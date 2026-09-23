"""Scripted Responses API outputs. The envelope is the minimum ChatOpenAI
accepts (verified fact 6 in the Plan 2 document)."""
from __future__ import annotations

import json
import threading
import time
import uuid


def message(text: str) -> dict:
    return {"type": "message", "id": f"msg_{uuid.uuid4().hex[:12]}", "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


def function_call(name: str, args: dict, call_id: str) -> dict:
    return {"type": "function_call", "id": f"fc_{uuid.uuid4().hex[:12]}", "call_id": call_id,
            "name": name, "arguments": json.dumps(args), "status": "completed"}


class MockBook:
    def __init__(self) -> None:
        self._scripts: dict[str, list[list[dict]]] = {}
        self._cursor: dict[tuple[str, str], int] = {}
        self._lock = threading.Lock()

    @classmethod
    def default(cls) -> "MockBook":
        book = cls()
        book.add("smoke", [[message("ok")]])
        return book

    def add(self, name: str, outputs: list[list[dict]]) -> None:
        if not outputs:
            raise ValueError("a mock script needs at least one response")
        self._scripts[name] = outputs

    def next(self, name: str, token: str) -> list[dict]:
        if name.startswith("final:"):
            return [message(name[len("final:"):])]
        script = self._scripts.get(name)
        if script is None:
            raise KeyError(f"no mock script {name!r}")
        with self._lock:
            i = self._cursor.get((name, token), 0)
            self._cursor[(name, token)] = i + 1
        return script[min(i, len(script) - 1)]

    @staticmethod
    def response(model: str, output: list[dict]) -> dict:
        return {"id": f"resp_{uuid.uuid4().hex}", "object": "response",
                "created_at": int(time.time()), "status": "completed", "model": model,
                "output": output, "parallel_tool_calls": True, "tool_choice": "auto",
                "tools": [],
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                          "input_tokens_details": {"cached_tokens": 0},
                          "output_tokens_details": {"reasoning_tokens": 0}}}
