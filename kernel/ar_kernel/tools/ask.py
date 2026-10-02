"""ask: one stateless question to the agent's model, with optional images.

The only way an image reaches the model: the gateway refuses images in agent conversations, and
this tool downscales each image and counts it against the caller's allowance (per phase attempt).
The call is recorded and charged to the run budget like a gateway call."""
from __future__ import annotations

import asyncio
import base64
import io
import threading
import time
from typing import Annotated, Any

from mcp.server.mcpserver import Context
from PIL import Image
from pydantic import Field

from .captioner import clip_host_path
from .context import PathError
from .server import ToolError

ENDPOINT = "/v1/chat/completions"
MOCK_ANSWER = "mock answer"


class Ask:
    def __init__(self, cfg, upstream, store, budget, model: str) -> None:
        self.upstream, self.store, self.budget, self.model = upstream, store, budget, model
        self.per_phase = int(cfg.get("ask.max_images_per_phase"))
        self.per_call = int(cfg.get("ask.max_images_per_call"))
        self.max_side = int(cfg.get("ask.max_image_side"))
        self._used: dict[str, int] = {}             # caller token -> images sent
        self._lock = threading.Lock()

    @property
    def description(self) -> str:
        return ("Ask the model that runs you one question and get its answer. Every call stands alone: the "
                "model sees only `question` and `images`, nothing of your conversation. `images`: at most "
                f"{self.per_call} image files under /workspace per call (relative paths resolve against "
                f"/workspace), each downscaled to at most {self.max_side} px on its longer side. At most "
                f"{self.per_phase} images in this phase; a question without images is not limited. "
                "Returns {answer, images_left}.")

    def _image(self, caller, path: str) -> dict:
        try:
            with Image.open(clip_host_path(caller, path)) as im:
                im = im.convert("RGB")
                im.thumbnail((self.max_side, self.max_side))
                buffer = io.BytesIO()
                im.save(buffer, "JPEG", quality=90)
        except PathError as exc:
            raise ToolError(f"{path}: {exc}") from exc
        except Exception as exc:                    # noqa: BLE001 -- PIL raises many types on a bad file
            raise ToolError(f"{path} cannot be read as an image: {type(exc).__name__}: {exc}") from exc
        data = base64.b64encode(buffer.getvalue()).decode()
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}

    def ask(self, caller, question: str, images: list[str]) -> dict:
        if not question.strip():
            raise ToolError("question is empty")
        if len(images) > self.per_call:
            raise ToolError(f"at most {self.per_call} images per call: got {len(images)}")
        parts = [{"type": "text", "text": question}, *[self._image(caller, p) for p in images]]
        why = self.budget.exhausted()
        if why:
            raise ToolError(why)
        with self._lock:
            used = self._used.get(caller.token, 0)
            if used + len(images) > self.per_phase:
                raise ToolError(f"{self.per_phase - used} of the {self.per_phase} images of this phase are left: "
                                f"got {len(images)}")
            self._used[caller.token] = used + len(images)
        body = self.upstream.enforce("/chat/completions", {
            "model": self.model, "messages": [{"role": "user", "content": parts}]})
        meta = self.store.begin(caller, ENDPOINT, body, tool="ask")
        started = time.monotonic()
        status, payload, attempts = asyncio.run(self.upstream.post("/chat/completions", body))
        usage = payload.get("usage") if isinstance(payload, dict) else None
        self.store.end(meta, caller, status=status, body=payload, latency_s=time.monotonic() - started,
                       attempts=attempts, cost_usd=self.budget.record(status, usage))
        try:
            answer = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            answer = None
        if status != 200 or not answer:
            with self._lock:                        # a failed call costs no images
                self._used[caller.token] -= len(images)
            raise ToolError(f"the model call failed (HTTP {status}): {str(payload)[:2000]}")
        return {"answer": answer, "images_left": self.per_phase - self._used[caller.token]}


class MockAsk:
    """ask in contract smoke runs: no model call."""
    description = "Ask the model one question (mock)."

    def ask(self, caller, question: str, images: list[str]) -> dict:
        return {"answer": MOCK_ANSWER, "images_left": 0}


def register_ask_tool(mcp, kit, ask) -> None:
    @mcp.tool(name="ask", description=ask.description)
    async def ask_tool(
            question: Annotated[str, Field(description="the whole question: the model sees nothing else of your conversation")],
            ctx: Context,
            images: Annotated[list[str] | None, Field(description="image files under /workspace to show with the question")] = None,
    ) -> dict[str, Any]:
        return await kit.call(ctx, "ask", {"question": question, "images": images or []},
                              lambda c: ask.ask(c, question, images or []))
