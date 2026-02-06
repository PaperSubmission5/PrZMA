# PrZMA/Automation_Agent/automation_agent.py
from __future__ import annotations

import argparse
import json
import os
import time
import ast
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections.abc import Mapping

import requests
import rpyc
from dotenv import load_dotenv

dotenv_path = Path(__file__).resolve().parents[1] / ".env"  # Automation_Agent/.. = PrZMA
load_dotenv(dotenv_path)

# Utils
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_now_ts() -> float:
    return time.time()


def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (bytes, bytearray)):
            return o.decode("utf-8", errors="replace")
        return str(o)

    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=default) + "\n")


def to_jsonable(x, _depth=0, _max_depth=8):
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="replace")

    if _depth >= _max_depth:
        return str(x)

    try:
        if hasattr(x, "items"):
            return {str(k): to_jsonable(v, _depth + 1) for k, v in x.items()}
    except Exception:
        pass

    if isinstance(x, (list, tuple, set)):
        return [to_jsonable(v, _depth + 1) for v in x]

    return str(x)


def _coerce_result_dict(res: Any) -> Dict[str, Any]:
    if res is None:
        return {}
    if isinstance(res, dict):
        return res
    if isinstance(res, Mapping):
        return dict(res)
    if isinstance(res, str):
        try:
            v = ast.literal_eval(res)
            if isinstance(v, dict):
                return v
        except Exception:
            return {}
    return {}

# Action Space (actions.json)
@dataclass
class ActionSpec:
    name: str
    summary: str
    params_schema: Dict[str, Any]


def load_action_specs(actions_path: Path) -> Dict[str, ActionSpec]:
    data = read_json(actions_path)
    specs: Dict[str, ActionSpec] = {}
    for a in data.get("actions", []):
        name = a["name"]
        specs[name] = ActionSpec(
            name=name,
            summary=a.get("summary", ""),
            params_schema=a.get("params_schema", {"type": "object", "properties": {}, "required": []}),
        )
    return specs


def validate_params(params: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    if schema.get("type") != "object":
        return errs

    props = schema.get("properties", {}) or {}
    required = schema.get("required", []) or []
    additional = schema.get("additionalProperties", True)

    for r in required:
        if r not in params:
            errs.append(f"missing required param: '{r}'")

    if additional is False:
        for k in params.keys():
            if k not in props:
                errs.append(f"unknown param not allowed: '{k}'")

    for k, v in params.items():
        ps = props.get(k)
        if not ps:
            continue
        t = ps.get("type")
        if t == "string" and not isinstance(v, str):
            errs.append(f"param '{k}' must be string")
        elif t == "integer" and not isinstance(v, int):
            errs.append(f"param '{k}' must be integer")
        elif t == "number" and not isinstance(v, (int, float)):
            errs.append(f"param '{k}' must be number")
        elif t == "boolean" and not isinstance(v, bool):
            errs.append(f"param '{k}' must be boolean")
        elif t == "object" and not isinstance(v, dict):
            errs.append(f"param '{k}' must be object")
        elif t == "array" and not isinstance(v, list):
            errs.append(f"param '{k}' must be array")

        if isinstance(v, str):
            if "minLength" in ps and len(v) < int(ps["minLength"]):
                errs.append(f"param '{k}' too short")
            if "maxLength" in ps and len(v) > int(ps["maxLength"]):
                errs.append(f"param '{k}' too long")

    return errs

# RPyC VM calls
def connect_vm(host: str, port: int, timeout: int = 180) -> rpyc.Connection:
    return rpyc.connect(
        host,
        port,
        config={
            "sync_request_timeout": timeout,
            "allow_public_attrs": True,
            "allow_all_attrs": True,
            "allow_pickle": True,
        },
    )


def vm_execute_action(conn: rpyc.Connection, req: Dict[str, Any]) -> Dict[str, Any]:
    payload = json.dumps(req, ensure_ascii=False)
    res = conn.root.execute_action(payload)
    return to_jsonable(res)


def vm_snapshot_collect(conn: rpyc.Connection, policy: Dict[str, Any]) -> Dict[str, Any]:
    payload = json.dumps(policy, ensure_ascii=False)
    res = conn.root.snapshot_collect(payload)
    return to_jsonable(res)

# OpenAI call
def openai_chat(model: str, api_key: str, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": messages, "temperature": temperature}
    r = requests.post(url, headers=headers, json=body, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text[:800]}")
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

# Action space per agent
def action_space_for_agent(specs: Dict[str, ActionSpec], agent_platforms: Any) -> List[ActionSpec]:
    enabled_namespaces = set()

    if isinstance(agent_platforms, dict):
        for plat, cfg in (agent_platforms or {}).items():
            if isinstance(cfg, dict) and cfg.get("enabled", False):
                ns = plat.split("_", 1)[0]
                enabled_namespaces.add(ns)

    elif isinstance(agent_platforms, list):
        for item in agent_platforms:
            if isinstance(item, str):
                enabled_namespaces.add(item.split("_", 1)[0])

    enabled_namespaces.add("browser")
    enabled_namespaces.add("web")

    out = []
    for s in specs.values():
        ns = s.name.split(".", 1)[0]
        if ns in enabled_namespaces:
            out.append(s)
    out.sort(key=lambda x: x.name)
    return out


def format_action_space(spec_list: List[ActionSpec], max_items: int = 120) -> str:
    items = spec_list[:max_items]
    lines = []
    for s in items:
        lines.append(f"- {s.name}: {s.summary}")
        lines.append(f"  params_schema: {json.dumps(s.params_schema, ensure_ascii=False)}")
    return "\n".join(lines)

# Prompts
def build_agent_system_prompt(config: Dict[str, Any], agent_id: str, spec_list: List[ActionSpec]) -> str:
    scenario = (config.get("scenario") or {}).get("objective", "")
    global_prompt = config.get("global_prompt") or {}
    interaction_style = global_prompt.get("interaction_style", [])
    hard_constraints = global_prompt.get("hard_constraints", [])
    completion_criteria = global_prompt.get("completion_criteria", [])

    agent_cfg = (config.get("agents") or {}).get(agent_id) or {}

    persona_raw = agent_cfg.get("persona")
    persona_text = ""
    tone = ""
    rules = []

    if isinstance(persona_raw, dict):
        tone = str(persona_raw.get("tone", ""))
        rules = persona_raw.get("behavior_rules", []) or []
        persona_text = str(persona_raw.get("text") or persona_raw.get("prompt") or "")
    elif isinstance(persona_raw, str):
        persona_text = persona_raw

    run_limits = config.get("run_limits") or {}
    max_actions_total = run_limits.get("max_actions_total")
    max_minutes = run_limits.get("max_minutes")

    return (
        "You are an automation agent controlling ONE VM-bound persona.\n"
        f"Agent: {agent_id}\n\n"
        "High-level scenario objective:\n"
        f"{scenario}\n\n"
        "Global interaction_style:\n"
        f"{json.dumps(interaction_style, ensure_ascii=False)}\n\n"
        "Hard constraints:\n"
        f"{json.dumps(hard_constraints, ensure_ascii=False)}\n\n"
        "Completion criteria (human text):\n"
        f"{json.dumps(completion_criteria, ensure_ascii=False)}\n\n"
        "Run limits (hard stops):\n"
        f"{json.dumps({'max_actions_total': max_actions_total, 'max_minutes': max_minutes}, ensure_ascii=False)}\n\n"
        "Your persona:\n"
        f"- persona_text: {persona_text}\n"
        f"- tone: {tone}\n"
        f"- behavior_rules: {json.dumps(rules, ensure_ascii=False)}\n\n"
        "You MUST choose your next action ONLY from the allowed Action Space below.\n"
        "When you choose an action, your params MUST match that action's params_schema EXACTLY.\n"
        "Do not invent param names.\n\n"
        "Output MUST be STRICT JSON with this schema:\n"
        "{\n"
        '  "action": {"name": "<string or null>", "params": <object>},\n'
        '  "reason": "<short reason>",\n'
        '  "done": <boolean>\n'
        "}\n"
        "If you truly cannot find any useful action that helps the objective or constraints, set done=true.\n"
        "If done=true, set action.name=null and action.params={}.\n\n"
        "Allowed Action Space:\n"
        f"{format_action_space(spec_list)}\n"
    )


def build_agent_user_prompt(step: int, observations: Dict[str, Any], progress: Dict[str, Any]) -> str:
    safe_obs = to_jsonable(observations)
    safe_prog = to_jsonable(progress)
    return (
        f"Step={step}\n"
        "Progress (JSON):\n"
        f"{json.dumps(safe_prog, ensure_ascii=False, indent=2)}\n\n"
        "Observations (JSON):\n"
        f"{json.dumps(safe_obs, ensure_ascii=False, indent=2)}\n\n"
        "Decide the NEXT single action for your agent.\n"
    )

# LLM decision per agent (validate, re-ask if invalid)
def llm_choose_next_action(
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    spec_map: Dict[str, ActionSpec],
    max_repairs: int = 2,
) -> Dict[str, Any]:
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    last_err = None

    for _ in range(max_repairs + 1):
        out = openai_chat(model=model, api_key=api_key, messages=messages)

        try:
            obj = json.loads(out)
        except Exception:
            last_err = "Output is not valid JSON."
            messages.append({"role": "assistant", "content": out})
            messages.append({"role": "user", "content": f"STRICT JSON only. Error: {last_err}\nRe-output ONLY JSON."})
            continue

        if not isinstance(obj, dict) or "done" not in obj or "action" not in obj or "reason" not in obj:
            last_err = "JSON must include keys: done, action, reason."
            messages.append({"role": "assistant", "content": out})
            messages.append({"role": "user", "content": f"Schema error: {last_err}\nRe-output ONLY JSON."})
            continue

        action = obj.get("action") or {}
        name = action.get("name")
        params = action.get("params") or {}

        if obj.get("done") is True:
            obj["action"] = {"name": None, "params": {}}
            if not isinstance(obj.get("reason"), str):
                obj["reason"] = str(obj.get("reason"))
            return obj

        if not isinstance(name, str) or name not in spec_map:
            last_err = f"Action name must be one of allowed actions. got={name}"
            messages.append({"role": "assistant", "content": out})
            messages.append({"role": "user", "content": f"{last_err}\nChoose ONLY from action space. Re-output ONLY JSON."})
            continue

        if not isinstance(params, dict):
            last_err = "action.params must be an object"
            messages.append({"role": "assistant", "content": out})
            messages.append({"role": "user", "content": f"{last_err}\nRe-output ONLY JSON."})
            continue

        schema = spec_map[name].params_schema
        errs = validate_params(params, schema)
        if errs:
            last_err = "Params do not match schema: " + "; ".join(errs)
            messages.append({"role": "assistant", "content": out})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"{last_err}\n"
                        f"Action '{name}' params_schema is:\n{json.dumps(schema, ensure_ascii=False)}\n"
                        "Re-output ONLY JSON with corrected params."
                    ),
                }
            )
            continue

        if not isinstance(obj.get("reason"), str):
            obj["reason"] = str(obj.get("reason"))

        return obj

    raise RuntimeError(f"LLM failed to produce valid action after repairs. last_err={last_err}")

# Logging helper (count actions consistently)
def log_action(
    action_log_path: Path,
    *,
    run_id: str,
    agent_id: str,
    name: Optional[str],
    params: Dict[str, Any],
    reason: str,
    result: Dict[str, Any],
    kind: str = "action",
) -> None:
    entry = {
        "ts": now_iso(),
        "run_id": run_id,
        "agent_id": agent_id,
        "kind": kind,  # "bootstrap" | "action" | "done" | "error" (optional use)
        "action": {"name": name, "params": params, "reason": reason},
        "result": to_jsonable(result),
    }
    append_jsonl(action_log_path, entry)

# Bootstrap actions (counted as actions)
def bootstrap_agent(
    run_id: str,
    agent_id: str,
    agent_cfg: Dict[str, Any],
    rendezvous: Dict[str, Any],
    conn: rpyc.Connection,
    action_specs: Dict[str, ActionSpec],
    action_log_path: Path,
) -> int:
    """
    Minimal boot:
    - browser.launch
    - if discord_web: discord.open, discord.login (optional), discord.goto_channel
    Returns: number of executed actions (for counting).
    """
    executed = 0

    # browser.launch
    if "browser.launch" in action_specs:
        bcfg = agent_cfg.get("browser") or {}
        req = {
            "schema_version": "1.0.0",
            "run_id": run_id,
            "agent_id": agent_id,
            "action_id": f"act_{agent_id}_{int(time.time()*1000)}",
            "name": "browser.launch",
            "params": {
                "browser_config": {
                    "browser": (bcfg.get("engine") or "chromium"),
                    "channel": (bcfg.get("channel") or "chrome"),
                    "headless": bool(bcfg.get("headless", False)),
                    "user_data_dir": bcfg.get("user_data_dir"),
                    "profile_name": (bcfg.get("profile_name") or "Default"),
                    "locale": (bcfg.get("locale") or "en-US"),
                    "timezone": (bcfg.get("timezone") or "UTC"),
                    "extra_args": list(bcfg.get("extra_args") or []),
                }
            },
        }
        res = vm_execute_action(conn, req)
        log_action(
            action_log_path,
            run_id=run_id,
            agent_id=agent_id,
            name=req["name"],
            params=req["params"],
            reason="bootstrap: launch browser",
            result=res,
            kind="bootstrap",
        )
        executed += 1

    platforms = agent_cfg.get("platforms") or {}
    rendezvous_platform = (rendezvous or {}).get("platform")

    if rendezvous_platform == "discord_web":
        dcfg = platforms.get("discord_web") or {}
        if isinstance(dcfg, dict) and dcfg.get("enabled", False):
            # discord.open
            if "discord.open" in action_specs:
                req = {
                    "schema_version": "1.0.0",
                    "run_id": run_id,
                    "agent_id": agent_id,
                    "action_id": f"act_{agent_id}_{int(time.time()*1000)}",
                    "name": "discord.open",
                    "params": {},
                }
                res = vm_execute_action(conn, req)
                log_action(
                    action_log_path,
                    run_id=run_id,
                    agent_id=agent_id,
                    name=req["name"],
                    params=req["params"],
                    reason="bootstrap: open discord",
                    result=res,
                    kind="bootstrap",
                )
                executed += 1

            # discord.login
            if dcfg.get("login_required", False) and "discord.login" in action_specs:
                cred_ref = dcfg.get("credential_ref")
                if not cred_ref:
                    raise RuntimeError(f"[{agent_id}] discord_web.login_required=true but credential_ref missing")

                email = os.getenv(f"{cred_ref}_EMAIL")
                pwd = os.getenv(f"{cred_ref}_PASSWORD")
                if not email or not pwd:
                    raise RuntimeError(f"Missing env vars for {cred_ref}: {cred_ref}_EMAIL / {cred_ref}_PASSWORD")

                req = {
                    "schema_version": "1.0.0",
                    "run_id": run_id,
                    "agent_id": agent_id,
                    "action_id": f"act_{agent_id}_{int(time.time()*1000)}",
                    "name": "discord.login",
                    "params": {"email": email, "password": pwd},
                }
                res = vm_execute_action(conn, req)
                log_action(
                    action_log_path,
                    run_id=run_id,
                    agent_id=agent_id,
                    name=req["name"],
                    params={"email": "***", "password": "***"},
                    reason="bootstrap: login discord",
                    result=res,
                    kind="bootstrap",
                )
                executed += 1

            # discord.goto_channel
            if "discord.goto_channel" in action_specs:
                channel_url_var = "DISCORD_MEETING_CHANNEL"
                channel_url = os.getenv(channel_url_var)
                if not channel_url:
                    raise RuntimeError(f"Missing env var: {channel_url_var} (Discord channel URL)")

                req = {
                    "schema_version": "1.0.0",
                    "run_id": run_id,
                    "agent_id": agent_id,
                    "action_id": f"act_{agent_id}_{int(time.time()*1000)}",
                    "name": "discord.goto_channel",
                    "params": {"channel_url": channel_url},
                }
                res = vm_execute_action(conn, req)
                log_action(
                    action_log_path,
                    run_id=run_id,
                    agent_id=agent_id,
                    name=req["name"],
                    params=req["params"],
                    reason="bootstrap: enter rendezvous channel",
                    result=res,
                    kind="bootstrap",
                )
                executed += 1

    return executed

# Fetch latest messages (Discord)

def fetch_latest_messages_if_supported(
    run_id: str,
    agent_id: str,
    conn: rpyc.Connection,
    action_specs: Dict[str, ActionSpec],
    limit: int = 10,
) -> Optional[Dict[str, Any]]:
    if "discord.get_latest_messages" not in action_specs:
        return None

    req = {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "agent_id": agent_id,
        "action_id": f"act_{agent_id}_{int(time.time()*1000)}",
        "name": "discord.get_latest_messages",
        "params": {"limit": limit},
    }
    res = vm_execute_action(conn, req)
    if res.get("ok") is True:
        return res.get("outputs")
    return {"error": res.get("error"), "outputs": res.get("outputs", {})}


# Run limits / termination

def read_run_limits(config: Dict[str, Any]) -> Dict[str, Any]:
    rl = config.get("run_limits") or {}
    # defaults: safe but permissive
    max_actions_total = rl.get("max_actions_total")
    max_minutes = rl.get("max_minutes")

    # normalize
    if max_actions_total is None:
        max_actions_total = 30
    try:
        max_actions_total = int(max_actions_total)
    except Exception:
        max_actions_total = 30

    if max_minutes is None:
        max_minutes = 15
    try:
        max_minutes = int(max_minutes)
    except Exception:
        max_minutes = 15

    return {"max_actions_total": max_actions_total, "max_minutes": max_minutes}


def hard_stop_reached(start_ts: float, max_minutes: int) -> bool:
    if max_minutes <= 0:
        return False
    return (utc_now_ts() - start_ts) >= (max_minutes * 60)

def finalize_run_with_snapshot_trigger(
    *,
    run_id: str,
    agent_ids: List[str],
    conns: Dict[str, rpyc.Connection],
    per_agent_specmap: Dict[str, Dict[str, ActionSpec]],
    action_log_path: Path,
) -> None:
    """
    Always execute a final snapshot-triggering action so SnapshotEngine rules fire
    even when the loop ends by 'done' or run limits.
    Default trigger: browser.close (safe / idempotent).
    """
    for aid in agent_ids:
        # only do if this agent supports browser.close in action space
        if "browser.close" not in per_agent_specmap.get(aid, {}):
            continue

        req = {
            "schema_version": "1.0.0",
            "run_id": run_id,
            "agent_id": aid,
            "action_id": f"act_{aid}_{int(time.time()*1000)}",
            "name": "browser.close",
            "params": {},
        }

        try:
            res = vm_execute_action(conns[aid], req)
        except Exception as e:
            res = {"ok": False, "error": f"finalize browser.close failed: {e}"}

        log_action(
            action_log_path,
            run_id=run_id,
            agent_id=aid,
            name="browser.close",
            params={},
            reason="finalize: force snapshot trigger on run termination (browser.close)",
            result=res,
            kind="finalize", 
        )



# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="przma_config.json or interpreted_przma_config.json")
    ap.add_argument("--endpoints", required=True, help="Snapshot_Engine/vm_endpoints.json")
    ap.add_argument("--actions", required=True, help="shared/actions.json")
    ap.add_argument("--action-log", required=True, help="runs/.../actions.jsonl")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--latest-limit", type=int, default=10)
    args = ap.parse_args()

    # env
    load_dotenv(".env")  # Based on repo root
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing env var: OPENAI_API_KEY")

    model = os.getenv("OPENAI_MODEL", "gpt-5.2-thinking")

    config_path = Path(args.config)
    config = read_json(config_path)

    purpose = config.get("purpose")
    if purpose not in ("education", "tool_testing"):
        raise RuntimeError("config.purpose must be 'education' or 'tool_testing'")

    agents = config.get("agents") or {}
    agent_ids = list(agents.keys())
    if not agent_ids:
        raise RuntimeError("No agents in config")

    rendezvous = config.get("rendezvous") or {}
    action_log_path = Path(args.action_log)

    run_limits = read_run_limits(config)
    max_actions_total = run_limits["max_actions_total"]
    max_minutes = run_limits["max_minutes"]

    # action space
    specs_all = load_action_specs(Path(args.actions))

    # endpoints
    endpoints_data = read_json(Path(args.endpoints))
    endpoints = endpoints_data.get("endpoints", [])
    ep_map: Dict[str, Tuple[str, int]] = {}
    for e in endpoints:
        ep_map[e["agent_id"]] = (e["host"], int(e["port"]))

    # connect
    conns: Dict[str, rpyc.Connection] = {}
    for aid in agent_ids:
        if aid not in ep_map:
            raise RuntimeError(f"vm_endpoints.json missing mapping for agent_id={aid}")
        host, port = ep_map[aid]
        conns[aid] = connect_vm(host, port, timeout=120)
        if conns[aid].root.ping() != "pong":
            raise RuntimeError(f"VM_Agent not responding: {aid} {host}:{port}")

    # per-agent specs/prompts
    per_agent_specmap: Dict[str, Dict[str, ActionSpec]] = {}
    per_agent_system: Dict[str, str] = {}

    for aid in agent_ids:
        agent_cfg = agents.get(aid) or {}
        plats = agent_cfg.get("platforms") or agent_cfg.get("capabilities") or {}
        spec_list = action_space_for_agent(specs_all, plats)
        per_agent_specmap[aid] = {s.name: s for s in spec_list}
        per_agent_system[aid] = build_agent_system_prompt(config, aid, spec_list)

    # track agent done state
    agent_done: Dict[str, bool] = {aid: False for aid in agent_ids}

    # count total executed actions (bootstrap + main actions)
    actions_total = 0

    # start time
    start_ts = utc_now_ts()

    # Bootstrap (COUNTED)

    for aid in agent_ids:
        agent_cfg = agents.get(aid) or {}
        executed = bootstrap_agent(
            run_id=args.run_id,
            agent_id=aid,
            agent_cfg=agent_cfg,
            rendezvous=rendezvous,
            conn=conns[aid],
            action_specs={k: v for k, v in specs_all.items()},
            action_log_path=action_log_path,
        )
        actions_total += executed

        # hard stops can kick in even during bootstrap
        if actions_total >= max_actions_total:
            break
        if hard_stop_reached(start_ts, max_minutes):
            break

    # Main loop
    step = 0
    while True:
        # hard stops
        if actions_total >= max_actions_total:
            break
        if hard_stop_reached(start_ts, max_minutes):
            break

        # early stop if all agents are done
        if all(agent_done.values()):
            break

        progressed_this_round = False

        for aid in agent_ids:
            if actions_total >= max_actions_total:
                break
            if hard_stop_reached(start_ts, max_minutes):
                break

            if agent_done.get(aid) is True:
                continue  # already done

            # observations: latest messages + last actions excerpt
            #latest = fetch_latest_messages_if_supported(args.run_id, aid, conns[aid], specs_all, limit=args.latest_limit)
            latest = fetch_latest_messages_if_supported(args.run_id, aid, conns[aid], per_agent_specmap[aid], limit=args.latest_limit)
            
            recent: List[Dict[str, Any]] = []
            if action_log_path.exists():
                try:
                    lines = action_log_path.read_text(encoding="utf-8").splitlines()
                    for ln in lines[-20:]:
                        try:
                            recent.append(json.loads(ln))
                        except Exception:
                            pass
                except Exception:
                    pass

            observations = {
                "agent_id": aid,
                "recent_actions": recent,
                "discord_latest_messages": latest,
            }

            progress = {
                "actions_total": actions_total,
                "max_actions_total": max_actions_total,
                "minutes_elapsed": int((utc_now_ts() - start_ts) // 60),
                "max_minutes": max_minutes,
                "agent_done": agent_done,
            }

            sys_prompt = per_agent_system[aid]
            usr_prompt = build_agent_user_prompt(step=step, observations=observations, progress=progress)

            try:
                decision = llm_choose_next_action(
                    model=model,
                    api_key=api_key,
                    system_prompt=sys_prompt,
                    user_prompt=usr_prompt,
                    spec_map=per_agent_specmap[aid],
                    max_repairs=2,
                )
            except Exception as e:
                # mark agent done on repeated LLM failure? conservative: log and mark done
                log_action(
                    action_log_path,
                    run_id=args.run_id,
                    agent_id=aid,
                    name=None,
                    params={},
                    reason=f"LLM decision failed; marking agent done. err={type(e).__name__}: {e}",
                    result={"ok": False, "error": str(e)},
                    kind="error",
                )
                agent_done[aid] = True
                continue

            if decision.get("done") is True:
                agent_done[aid] = True
                log_action(
                    action_log_path,
                    run_id=args.run_id,
                    agent_id=aid,
                    name=None,
                    params={},
                    reason=decision.get("reason", "done"),
                    result={"ok": True, "note": "agent done"},
                    kind="done",
                )
                progressed_this_round = True
                continue

            action = decision["action"]
            name = action["name"]
            params = action.get("params") or {}

            if name == "browser.launch" and "browser_config" not in params:
                params = {"browser_config": params}

            if name not in per_agent_specmap[aid]:
                # should not happen; treat as done to avoid looping
                agent_done[aid] = True
                log_action(
                    action_log_path,
                    run_id=args.run_id,
                    agent_id=aid,
                    name=None,
                    params={},
                    reason=f"Invalid action name after validation: {name}. Marking done.",
                    result={"ok": False, "error": "invalid_action_name"},
                    kind="error",
                )
                progressed_this_round = True
                continue

            # Execute action
            req = {
                "schema_version": "1.0.0",
                "run_id": args.run_id,
                "agent_id": aid,
                "action_id": f"act_{aid}_{int(time.time()*1000)}",
                "name": name,
                "params": params,
            }
            res = vm_execute_action(conns[aid], req)

            log_action(
                action_log_path,
                run_id=args.run_id,
                agent_id=aid,
                name=name,
                params=params,
                reason=decision.get("reason", ""),
                result=res,
                kind="action",
            )
            actions_total += 1
            progressed_this_round = True

        step += 1

        # if no one progressed and not all done, avoid infinite loop
        if not progressed_this_round and not all(agent_done.values()):
            # log and break conservatively
            for aid in agent_ids:
                if not agent_done.get(aid):
                    log_action(
                        action_log_path,
                        run_id=args.run_id,
                        agent_id=aid,
                        name=None,
                        params={},
                        reason="No progress in round; stopping to avoid infinite loop.",
                        result={"ok": True, "note": "no_progress_stop"},
                        kind="done",
                    )
                    agent_done[aid] = True
            break
    # FINALIZE: always trigger snapshot once
    try:
        finalize_run_with_snapshot_trigger(
            run_id=args.run_id,
            agent_ids=agent_ids,
            conns=conns,
            per_agent_specmap=per_agent_specmap,
            action_log_path=action_log_path,
        )
    except Exception as e:
        # Even in the worst case, the run must end, so we swallow the exception and leave only the log
        for aid in agent_ids:
            log_action(
                action_log_path,
                run_id=args.run_id,
                agent_id=aid,
                name=None,
                params={},
                reason=f"finalize snapshot trigger failed: {e}",
                result={"ok": False, "error": str(e)},
                kind="error",
            )
    # final run summary (optional)
    summary = {
        "schema_version": "1.0.0",
        "run_id": args.run_id,
        "finished_at": now_iso(),
        "purpose": purpose,
        "actions_total": actions_total,
        "max_actions_total": max_actions_total,
        "minutes_elapsed": int((utc_now_ts() - start_ts) // 60),
        "max_minutes": max_minutes,
        "agent_done": agent_done,
        "stop_reason": (
            "max_actions_total"
            if actions_total >= max_actions_total
            else ("max_minutes" if hard_stop_reached(start_ts, max_minutes) else "all_agents_done")
        ),
    }
    # write beside action log
    try:
        write_json(action_log_path.with_suffix(".summary.json"), summary)
    except Exception:
        pass

    for aid, conn in conns.items():
        try:
            conn.root.close_agent(aid)
        except Exception:
            pass

    # close conns
    for c in conns.values():
        try:
            c.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
