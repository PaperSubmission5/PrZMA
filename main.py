# PrZMA/main.py
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path
from typing import Any, Dict
import signal
import time


# Include PrZMA Root in import path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Automation_Agent.agent_generator import generate_vm_endpoints


def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))

def write_json(p:Path, obj:Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="PrZMA/przma_config.json")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--catalog", required=True, help="PrZMA/Snapshot_Engine/artifact_catalog.json")
    ap.add_argument("--out-dir", required=True, help="PrZMA/runs")
    ap.add_argument("--max-steps", type=int, default=10)

    # Since rules.json is loaded once when the engine starts, it is created/confirmed in advance before running the engine in main
    ap.add_argument("--rules", default=str(ROOT / "Snapshot_Engine" / "rules.json"))

    args = ap.parse_args()

    config_path = Path(args.config)
    config: Dict[str, Any] = read_json(config_path)

    # 0) write rules.json before starting engine
    rules_path = Path(args.rules)
    snapshot_rules = config.get("snapshot")
    if not isinstance(snapshot_rules, dict):
        raise RuntimeError("config.snapshot missing or not a dict")
    write_json(rules_path, snapshot_rules)

    agents = config.get("agents") or {}
    agent_ids = list(agents.keys())
    if not agent_ids:
        raise RuntimeError("No agents in config")

    # run dir + action log
    out_dir = Path(args.out_dir)
    run_dir = out_dir / f"run_{args.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    action_log = run_dir / "actions.jsonl"

    # 1) generate vm_endpoints.json via agent_generator
    endpoints_path = ROOT / "Snapshot_Engine" / "vm_endpoints.json"
    generate_vm_endpoints(
        config=config,
        agent_ids=agent_ids,
        endpoints_path=endpoints_path,
        boot_first=True,
        wait_timeout_sec=int((config.get("vm_boot") or {}).get("wait_timeout_sec", 180)),
        poll_interval_sec=float((config.get("vm_boot") or {}).get("poll_interval_sec", 2.0)),
    )

    # 2) start Snapshot_Engine (tail action log)
    engine_py = ROOT / "Snapshot_Engine" / "engine.py"
    engine_cmd = [
        sys.executable,
        str(engine_py),
        "--run-id",
        args.run_id,
        "--rules",
        str(Path(args.rules)),
        "--catalog",
        str(Path(args.catalog)),
        "--endpoints",
        str(endpoints_path),
        "--out",
        str(out_dir),
        "--action-log",
        str(action_log),
        "--drain-timeout-sec", "600",
    ]
    engine_p = subprocess.Popen(
        engine_cmd,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
    )


    # 3) run Automation_Agent
    agent_py = ROOT / "Automation_Agent" / "automation_agent.py"
    actions_json = ROOT / "shared" / "actions.json"

    agent_cmd = [
        sys.executable,
        str(agent_py),
        "--config",
        str(config_path),
        "--endpoints",
        str(endpoints_path),
        "--actions",
        str(actions_json),
        "--action-log",
        str(action_log),
        "--run-id",
        args.run_id,
    ]
    rc = subprocess.call(agent_cmd)

    # 4) shutdown engine (graceful -> terminate -> force kill)
    try:
        # Graceful: ask engine to stop + drain
        engine_p.send_signal(signal.CTRL_BREAK_EVENT)
        engine_p.wait(timeout=900)

    except Exception:
        # If graceful failed, try terminate
        try:
            engine_p.terminate()
            engine_p.wait(timeout=300)
        except Exception:
            pass

        # Force kill process tree (if still alive)
        try:
            if engine_p.poll() is None:
                subprocess.run(
                    ["taskkill", "/PID", str(engine_p.pid), "/T", "/F"],
                    capture_output=True,
                    text=True,
                )
        except Exception:
            pass


if __name__ == "__main__":
    main()
