"""ask: one stateless question to the agent's model, with capped, downscaled images."""
import base64
import io
import json

import httpx
import pytest
from PIL import Image

from ar_kernel.budget import Budget
from ar_kernel.config import KernelConfig
from ar_kernel.gateway.store import CallStore
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.tools.ask import Ask
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.tools.server import ToolError
from tests.test_gateway_app import _upstream_that

REAL = KernelConfig.load()


def _answer(request):
    return httpx.Response(200, json={"id": "c", "choices": [{"message": {"role": "assistant", "content": "a cat"}}],
                                     "usage": {"prompt_tokens": 10, "completion_tokens": 2}})


@pytest.fixture
def make(tmp_path):
    def build(handler=_answer, budget=None, **limits):
        cfg = KernelConfig(raw={**REAL.raw, "ask": {**REAL.raw["ask"], **limits}}, repo_root=REAL.repo_root)
        rec = Recorder(tmp_path / "run")
        caller = TokenRegistry(rec).issue(node="n1", phase="edit_self", attempt=1, workspace_host=tmp_path,
                                          staging_host=tmp_path / "staging")
        seen = []
        ask = Ask(cfg, _upstream_that(handler, seen), CallStore(rec), budget or Budget(), "gpt-x")
        Image.new("RGB", (3000, 1500), (200, 30, 30)).save(tmp_path / "big.png")
        return ask, caller, seen, rec
    return build


def test_a_question_alone_is_one_user_message_and_costs_no_images(make):
    ask, caller, seen, _ = make()
    assert ask.ask(caller, "what is 2+2?", []) == {"answer": "a cat", "images_left": REAL.get("ask.max_images_per_phase")}
    assert seen == [{"model": "gpt-x", "messages": [{"role": "user", "content": [{"type": "text", "text": "what is 2+2?"}]}]}]


def test_images_are_downscaled_and_counted(make):
    ask, caller, seen, _ = make(max_image_side=512, max_images_per_phase=3)
    assert ask.ask(caller, "what is this?", ["big.png", "/workspace/big.png"])["images_left"] == 1
    parts = seen[0]["messages"][0]["content"][1:]
    assert len(parts) == 2
    for part in parts:
        kind, data = part["image_url"]["url"].split(",", 1)
        assert kind == "data:image/jpeg;base64"
        assert Image.open(io.BytesIO(base64.b64decode(data))).size == (512, 256)
    with pytest.raises(ToolError, match="1 of the 3 images"):
        ask.ask(caller, "and these?", ["big.png", "big.png"])
    assert ask.ask(caller, "and this?", ["big.png"])["images_left"] == 0
    assert ask.ask(caller, "no image", [])["images_left"] == 0        # text stays unlimited


def test_too_many_images_in_one_call_are_refused(make):
    ask, caller, seen, _ = make(max_images_per_call=2)
    with pytest.raises(ToolError, match="at most 2 images per call"):
        ask.ask(caller, "q", ["big.png"] * 3)
    assert seen == []


@pytest.mark.parametrize("path, match", [("/etc/passwd", "outside /workspace"), ("missing.png", "not a file"),
                                         ("notes.txt", "cannot be read as an image")])
def test_a_bad_image_path_is_refused(make, tmp_path, path, match):
    ask, caller, seen, _ = make()
    (tmp_path / "notes.txt").write_text("not an image")
    with pytest.raises(ToolError, match=match):
        ask.ask(caller, "q", [path])
    assert seen == []


def test_an_empty_question_is_refused(make):
    ask, caller, _, _ = make()
    with pytest.raises(ToolError, match="question is empty"):
        ask.ask(caller, "  ", [])


def test_a_failed_call_costs_no_images(make):
    ask, caller, _, _ = make(handler=lambda r: httpx.Response(400, json={"error": {"message": "no vision"}}),
                             max_images_per_phase=1)
    with pytest.raises(ToolError, match="HTTP 400.*no vision"):
        ask.ask(caller, "q", ["big.png"])
    ask.upstream = _upstream_that(_answer, [])
    assert ask.ask(caller, "q", ["big.png"])["images_left"] == 0


def test_the_call_is_recorded_and_charged_like_a_gateway_call(make):
    budget = Budget(prices={"input": 1.0, "cached_input": 1.0, "output": 1.0})
    ask, caller, _, rec = make(budget=budget)
    ask.ask(caller, "q", ["big.png"])
    request, response = rec.read_events("n1")
    assert (request["type"], response["type"]) == ("llm.request", "llm.response")
    assert request["phase"] == "edit_self" and response["status"] == 200 and response["cost_usd"] == 12 / 1e6
    assert budget.calls == 1 and budget.tokens == 12
    body = rec.load_payload(request["payload"])["body"]
    assert json.dumps(body).count("data:image/jpeg;base64") == 1


def test_a_spent_budget_refuses_the_call(make):
    budget = Budget(max_usd=0.0, prices={"input": 1.0, "cached_input": 1.0, "output": 1.0})
    ask, caller, seen, _ = make(budget=budget)
    with pytest.raises(ToolError, match="budget spent"):
        ask.ask(caller, "q", [])
    assert seen == []


def test_the_description_states_the_configured_limits(make):
    ask, *_ = make(max_images_per_phase=7, max_images_per_call=3, max_image_side=640)
    assert all(text in ask.description for text in ("At most 7 images", "at most 3 image files", "640 px"))


@pytest.mark.manual
def test_live_ask_over_the_tool_socket(tmp_path):
    """pytest tests/test_ask.py -m manual -s
    The real path, with the model of .env: container client -> tool socket -> Ask -> provider."""
    import asyncio
    from fastapi import FastAPI
    from PIL import ImageDraw
    from ar_contract.client import mcp_session
    from ar_kernel.gateway.app import Upstream
    from ar_kernel.services import RunServices, socket_dir_for
    from ar_kernel.tools.ask import register_ask_tool
    from ar_kernel.tools.server import ToolKit, build_tool_app, new_mcp

    env = dict(line.strip().split("=", 1) for line in (REAL.repo_root / ".env").read_text().splitlines()
               if "=" in line and not line.startswith("#"))
    rec = Recorder(tmp_path / "run")
    reg = TokenRegistry(rec)
    caller = reg.issue(node="n1", phase="edit_self", attempt=1, workspace_host=tmp_path,
                       staging_host=tmp_path / "staging")
    mcp = new_mcp()
    register_ask_tool(mcp, ToolKit(reg, rec), Ask(REAL, Upstream.from_env(env, timeout_s=600, retries=2),
                                                  CallStore(rec), Budget(), env["OPENAI_MODEL"]))
    im = Image.new("RGB", (2000, 2000), "white")
    draw = ImageDraw.Draw(im)
    draw.rectangle([200, 200, 900, 900], fill="red")
    draw.ellipse([1100, 1100, 1800, 1800], fill="blue")
    im.save(tmp_path / "shapes.png")
    services = RunServices(socket_dir_for(tmp_path / "run"))
    services.start(FastAPI(), build_tool_app(mcp))

    async def calls():
        async with mcp_session(services.socket_dir, caller.token) as session:
            tools = {t.name: t for t in (await session.list_tools()).tools}
            text = await session.call_tool("ask", {"question": "What is 17 + 25? Answer with the number only."})
            image = await session.call_tool("ask", {"question": "Name the two shapes and their colours.",
                                                    "images": ["shapes.png"]})
            return tools["ask"].description, text, image
    try:
        description, text, image = asyncio.run(calls())
    finally:
        services.stop()
    print(description)
    for result in (text, image):
        assert not result.is_error, result.content
        print(json.dumps(result.structured_content))
    assert "42" in text.structured_content["answer"]
    assert text.structured_content["images_left"] == REAL.get("ask.max_images_per_phase")
    answer = image.structured_content["answer"].lower()
    assert all(word in answer for word in ("red", "blue", "square", "circle"))
    assert image.structured_content["images_left"] == REAL.get("ask.max_images_per_phase") - 1
