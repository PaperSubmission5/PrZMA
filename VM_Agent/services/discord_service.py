# VM_Agent/services/discord_service.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from playwright.sync_api import TimeoutError as PWTimeoutError

from .browser_service import BrowserService


class DiscordService:
    """
    Discord Web automation on top of BrowserService.
    - Recommended: run BrowserService with persistent profile so login persists.
    """

    def __init__(self, browser: BrowserService):
        self.browser = browser

    def _page(self, agent_id: str):
        return self.browser._page(agent_id)  

    def ensure_open(self, agent_id: str, timeout_ms: int = 30000) -> None:
        page = self._page(agent_id)
        if "discord.com" not in (page.url or ""):
            page.goto("https://discord.com/app", wait_until="domcontentloaded", timeout=timeout_ms)

    def action_open(self, agent_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        timeout_ms = int(params.get("timeout_ms", 30000))
        self.ensure_open(agent_id, timeout_ms)
        return {"current_url": self._page(agent_id).url}

    def action_login(self, agent_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Best-effort login. Prefer persistent profile in browser config.
        params:
          - email
          - password
        """
        email = params.get("email")
        password = params.get("password")
        timeout_ms = int(params.get("timeout_ms", 30000))

        page = self._page(agent_id)
        page.goto("https://discord.com/login", wait_until="domcontentloaded", timeout=timeout_ms)

        if email and password:
            page.locator("input[name='email']").first.fill(email, timeout=timeout_ms)
            page.locator("input[name='password']").first.fill(password, timeout=timeout_ms)
            page.locator("button[type='submit']").first.click(timeout=timeout_ms)

        # Wait for app to load (may be blocked by captcha/2FA)
        try:
            page.wait_for_url("**/app", timeout=timeout_ms)
        except Exception:
            pass

        return {"current_url": page.url, "note": "Login is best-effort; persistent profile is recommended."}

    def action_goto_channel(self, agent_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Most robust: provide channel URL.
        Accepts both:
          - url (canonical in actions.json)
          - channel_url (legacy)
        """
        timeout_ms = int(params.get("timeout_ms", 30000))
        self.ensure_open(agent_id, timeout_ms)
        page = self._page(agent_id)

        channel_url = params.get("url") or params.get("channel_url")
        if channel_url:
            page.goto(channel_url, wait_until="domcontentloaded", timeout=timeout_ms)
            return {"current_url": page.url}

        # Best-effort fallback (rely on Ctrl+K quick switcher)
        target = params.get("query") or params.get("channel_name")
        if not target:
            raise ValueError("discord.goto_channel requires url/channel_url or (query/channel_name).")

        try:
            page.keyboard.press("Control+K")
            box = page.locator(
                "input[aria-label*='Quick switcher'], input[placeholder*='Where would you like to go']"
            ).first
            box.fill(target, timeout=timeout_ms)
            page.keyboard.press("Enter")
            return {"switched": target, "current_url": page.url}
        except Exception as e:
            return {"current_url": page.url, "warning": f"Failed quick switcher: {e}"}

    def _find_message_box(self, page, timeout_ms: int):
        candidates = [
            "div[role='textbox'][data-slate-editor='true']",
            "div[role='textbox']",
            "textarea",
        ]
        for sel in candidates:
            loc = page.locator(sel).first
            try:
                loc.wait_for(timeout=2000)
                return loc
            except Exception:
                continue
        return None

    def action_send_message(self, agent_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        params:
          - text (required)
        """
        text = params["text"]
        timeout_ms = int(params.get("timeout_ms", 30000))
        self.ensure_open(agent_id, timeout_ms)
        page = self._page(agent_id)

        box = self._find_message_box(page, timeout_ms)
        if box is None:
            raise RuntimeError("Discord message box not found. Make sure you're inside a channel.")

        box.click(timeout=timeout_ms)
        box.type(text, timeout=timeout_ms)
        page.keyboard.press("Enter")
        return {"sent": True, "len": len(text), "current_url": page.url}

    def action_get_latest_messages(self, agent_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Best-effort fetch of recent visible messages from current channel UI.
        params:
          - limit (optional, default 10)
        returns:
          { "messages": [ { "author": str|null, "text": str, "ts": str|null } ... ] }
        """
        limit = int(params.get("limit", 10))
        timeout_ms = int(params.get("timeout_ms", 30000))
        self.ensure_open(agent_id, timeout_ms)
        page = self._page(agent_id)

        # Wait for message list to appear (best-effort)
        try:
            page.wait_for_selector("ol[data-list-id='chat-messages']", timeout=timeout_ms)
        except Exception:
            pass

        # Use DOM evaluation to be resilient to class name changes
        js = """
        (limit) => {
          const out = [];
          // Primary container used in Discord Web
          const root = document.querySelector("ol[data-list-id='chat-messages']") || document;
          // Try to collect message items (li) in the chat list
          const items = Array.from(root.querySelectorAll("li")).slice(-Math.max(1, limit) * 3); // oversample a bit
          for (const li of items) {
            // author (best-effort)
            const authorEl =
              li.querySelector("h3 span[role='button']") ||
              li.querySelector("span[class*='username']") ||
              li.querySelector("span[aria-label*='User']");

            const author = authorEl ? (authorEl.textContent || "").trim() : null;

            // message text: Discord often uses data-slate-node, or div[role="document"]
            const msgEl =
              li.querySelector("div[data-slate-node='value']") ||
              li.querySelector("div[role='document']") ||
              li.querySelector("div[class*='messageContent']");

            const text = msgEl ? (msgEl.textContent || "").trim() : "";
            if (!text) continue;

            // timestamp (best-effort)
            const timeEl = li.querySelector("time");
            const ts = timeEl ? (timeEl.getAttribute("datetime") || null) : null;

            out.push({ author, text, ts });
          }

          // keep last N
          return out.slice(-limit);
        }
        """
        messages = page.evaluate(js, limit)
        if not isinstance(messages, list):
            messages = []

        return {"current_url": page.url, "count": len(messages), "messages": messages}

    def action_upload_file(self, agent_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        params:
          - file_path (required)
          - text (optional)  # message to send with/after upload
        """
        file_path = params["file_path"]
        text = params.get("text")
        timeout_ms = int(params.get("timeout_ms", 30000))
        self.ensure_open(agent_id, timeout_ms)
        page = self._page(agent_id)

        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        # Locate and set files directly (Discord uses hidden input[type=file])
        input_loc = page.locator("input[type='file']").first
        try:
            input_loc.set_input_files(file_path, timeout=5000)
        except Exception:
            # Try clicking an upload/add button to reveal file input
            try:
                page.locator("button[aria-label*='Upload'], button[aria-label*='Add']").first.click(timeout=3000)
                page.locator("input[type='file']").first.set_input_files(file_path, timeout=5000)
            except Exception as e:
                raise RuntimeError(f"Discord file upload input not found: {e}")

        # If text provided, send it 
        if text:
            self.action_send_message(agent_id, {"text": text, "timeout_ms": timeout_ms})
        else:
            try:
                page.keyboard.press("Enter")
            except Exception:
                pass

        return {"uploaded": True, "file_path": file_path, "current_url": page.url}

    def execute(self, agent_id: str, action_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        verb = action_name.split(".", 1)[1] if "." in action_name else action_name

        if verb == "open":
            return self.action_open(agent_id, params)
        if verb == "login":
            return self.action_login(agent_id, params)
        if verb == "goto_channel":
            return self.action_goto_channel(agent_id, params)
        if verb == "send_message":
            return self.action_send_message(agent_id, params)
        if verb == "get_latest_messages":
            return self.action_get_latest_messages(agent_id, params)
        if verb == "upload_file":
            return self.action_upload_file(agent_id, params)

        raise ValueError(f"Unsupported discord action: {action_name}")
