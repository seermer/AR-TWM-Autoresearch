"""python -m ar_contract.run <edit_self|improve_recipe>  -- the only entry the kernel runs."""
from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import sys
import traceback

from .models import CONTEXT_MODELS, RESULT_MODELS


def _write(workspace: str, body: dict) -> None:
    path = os.path.join(workspace, "result.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(body, handle, default=str)
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    kind = argv[0] if argv else ""
    workspace = os.environ.get("AR_WORKSPACE", "/workspace")
    if kind not in CONTEXT_MODELS:
        print(f"usage: python -m ar_contract.run <{'|'.join(CONTEXT_MODELS)}>", file=sys.stderr)
        return 2
    try:
        context_dir = os.environ.get("AR_CONTEXT_DIR", "/context")
        with open(os.path.join(context_dir, "context.json"), encoding="utf-8") as handle:
            ctx = CONTEXT_MODELS[kind].model_validate(json.load(handle))
        sys.path.insert(0, os.environ.get("AR_AGENT_DIR", "/agent"))
        entry = importlib.import_module("agent.entry")
        value = getattr(entry, kind)(ctx)
        if inspect.isawaitable(value):
            async def _settle(awaitable):
                return await awaitable
            value = asyncio.run(_settle(value))      # coroutines and other awaitables alike
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        result = RESULT_MODELS[kind].model_validate(value)
    except Exception as exc:                  # noqa: BLE001 -- every failure becomes a result
        _write(workspace, {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                           "traceback": traceback.format_exc()})
        return 1
    _write(workspace, {"ok": True, "result": result.model_dump()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
