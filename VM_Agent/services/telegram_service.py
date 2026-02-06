# VM_Agent/services/telegram_service.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from playwright.sync_api import TimeoutError as PWTimeoutError

from .browser_service import BrowserService


class TelegramService:
    """
    Telegram Web automation (web.telegram.org).
    Recommended:
      - Use BrowserService with persistent profile (user_data_dir) and login once manually.
    """

    def __init__(self, browser: BrowserService):
        self.browser = browser

    def _page(self, agent_id: str):
        return self.browser._page(agent_id)

    def ensure_open(self, agent_id: str, variant: str = "k", timeout_ms: int = 30000) -> None:
        page = self._page(agent_id)
        base = "https://web.telegram.org"
        url = f"{base}/{variant}/"
        if "web.telegram.org" not in (page.url or ""):
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

    def action_open(self, agent_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        variant = (params.get("variant") or "k").lower()  # "k" or "a"
        timeout_ms = int(params.get("timeout_ms", 30000))
        self.ensure_open(agent_id, variant, timeout_ms)
        return {"current_url": self._page(agent_id).url, "variant": variant}

    def action_select_chat(self, agent_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        params:
          - chat: str (chat name / username)
        """
        chat = params["chat"]
        timeout_ms = int(params.get("timeout_ms", 30000))
        self.ensure_open(agent_id, (params.get("variant") or "k"), timeout_ms)
        page = self._page(agent_id)

        # Try search box
        candidates = [
            "input[placeholder*='Search']",
            "input[type='text']",
        ]
        box = None
        for sel in candidates:
            loc = page.locator(sel).first
            try:
                loc.wait_for(timeout=2000)
                box = loc
                break
            except Exception:
                continue

        if box is None:
            return {"warning": "Search box not found. Ensure Telegram Web is logged in and UI loaded.", "current_url": page.url}

        box.click(timeout=timeout_ms)
        box.fill(chat, timeout=timeout_ms)

        # click first result
        try:
            page.locator("div[role='listitem'], .ListItem, .chatlist a, .chatlist .row").first.click(timeout=timeout_ms)
        except Exception:
            # fallback (press enter)
            try:
                page.keyboard.press("Enter")
            except Exception:
                pass

        return {"selected": chat, "current_url": page.url}

    def action_send_message(self, agent_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        params:
          - chat (optional): if provided, will select chat first
          - text (required)
        """
        text = params["text"]
        timeout_ms = int(params.get("timeout_ms", 30000))
        self.ensure_open(agent_id, (params.get("variant") or "k"), timeout_ms)
        page = self._page(agent_id)

        if params.get("chat"):
            self.action_select_chat(agent_id, {"chat": params["chat"], "timeout_ms": timeout_ms, "variant": params.get("variant", "k")})

        # message input
        candidates = [
            "div[contenteditable='true']",
            "div[role='textbox']",
            "textarea",
        ]
        box = None
        for sel in candidates:
            loc = page.locator(sel).first
            try:
                loc.wait_for(timeout=2000)
                box = loc
                break
            except Exception:
                continue

        if box is None:
            raise RuntimeError("Telegram message box not found. Make sure a chat is open and you are logged in.")

        box.click(timeout=timeout_ms)
        box.type(text, timeout=timeout_ms)
        page.keyboard.press("Enter")
        return {"sent": True, "len": len(text), "current_url": page.url}

    def action_upload_file(self, agent_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        params:
          - file_path (required)
          - chat (optional)
          - message (optional)
        """
        file_path = params["file_path"]
        timeout_ms = int(params.get("timeout_ms", 30000))
        self.ensure_open(agent_id, (params.get("variant") or "k"), timeout_ms)
        page = self._page(agent_id)

        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        if params.get("chat"):
            self.action_select_chat(agent_id, {"chat": params["chat"], "timeout_ms": timeout_ms, "variant": params.get("variant", "k")})

        # Telegram uses hidden input[type=file] after clicking attach.
        try:
            # try direct
            page.locator("input[type='file']").first.set_input_files(file_path, timeout=5000)
        except Exception:
            # click attach icon (best-effort)
            try:
                page.locator("button[aria-label*='Attach'], button[title*='Attach'], .attach, .Button.Attach").first.click(timeout=3000)
                page.locator("input[type='file']").first.set_input_files(file_path, timeout=5000)
            except Exception as e:
                raise RuntimeError(f"Telegram file upload failed: {e}")

        if params.get("message"):
            self.action_send_message(agent_id, {"text": params["message"], "timeout_ms": timeout_ms, "variant": params.get("variant", "k")})
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
        if verb == "select_chat":
            return self.action_select_chat(agent_id, params)
        if verb == "send_message":
            return self.action_send_message(agent_id, params)
        if verb == "upload_file":
            return self.action_upload_file(agent_id, params)

        raise ValueError(f"Unsupported telegram action: {action_name}")
