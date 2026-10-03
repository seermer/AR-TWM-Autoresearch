import asyncio
import threading

from ar_kernel.contract.verify import _MockCaption, _MockData, _MockHf
from ar_kernel.telemetry.recorder import Recorder
from ar_kernel.tools.ask import MockAsk, register_ask_tool
from ar_kernel.tools.captioner import register_caption_tool
from ar_kernel.tools.context import TokenRegistry
from ar_kernel.tools.data_tools import register_data_tools
from ar_kernel.tools.hf_tools import register_hf_tools
from ar_kernel.tools.jobs import JobQueue, register_job_tools
from ar_kernel.tools.server import ToolKit, new_mcp
from ar_kernel.tools.skills import load_skills, register_skill_tool

SKILLS = {"container", "data_formats", "clip_quality", "retries", "node_files", "large_files"}


def _tools(tmp_path):
    rec = Recorder(tmp_path)
    queue = JobQueue(rec, threading.Lock(), wait_cap_s=5.0)
    try:
        kit, mcp = ToolKit(TokenRegistry(rec), rec), new_mcp()
        register_data_tools(mcp, kit, _MockData())
        register_hf_tools(mcp, kit, _MockHf())
        register_job_tools(mcp, kit, queue)
        queue.register(_MockCaption())
        register_caption_tool(mcp, kit, queue)
        register_ask_tool(mcp, kit, MockAsk())
        register_skill_tool(mcp, kit)
        return {t.name: t for t in asyncio.run(mcp.list_tools())}
    finally:
        queue.shutdown()


def test_every_skill_has_a_name_a_use_and_a_body():
    skills = load_skills()
    assert set(skills) == SKILLS
    for name, skill in skills.items():
        assert skill["description"].startswith("Use when") and len(skill["text"]) > 200, name
        assert "wbench" not in skill["text"].lower()
    assert "57 frames" in skills["data_formats"]["text"] and "cam_c2w" in skills["data_formats"]["text"]


def test_read_skill_lists_every_skill_in_its_description(tmp_path):
    tool = _tools(tmp_path)["read_skill"]
    for name, skill in load_skills().items():
        assert f"- {name}: {skill['description']}" in tool.description


def test_every_kernel_tool_argument_is_described(tmp_path):
    tools = _tools(tmp_path)
    assert "filter" not in tools["data_query"].input_schema["properties"]
    assert set(tools["data_query"].input_schema["properties"]) == {
        "format", "camera_motion", "clip_ids", "ingested_by", "limit", "offset"}
    for name, tool in tools.items():
        for arg, schema in tool.input_schema["properties"].items():
            assert schema.get("description"), f"{name}.{arg} has no description"
    candidate = tools["data_ingest"].input_schema["properties"]["candidates"]["anyOf"][0]["items"]
    assert set(candidate["required"]) == {"video", "caption", "camera_motion", "provenance"}
    assert all(p.get("description") for p in candidate["properties"].values())
