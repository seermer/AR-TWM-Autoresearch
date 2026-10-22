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
from ar_kernel.tools.knowledge import load_knowledge, register_knowledge_tool

FILES = {"container", "data_formats", "clip_quality", "node_files", "large_files", "training_text"}
AGENT_TEXT_NEVER_SAYS = ("window", "per_chunk", "prompt_mode", "camera_motion", "provenance", "license", "round grid",
                         "rounds", "close-up", "clearly visible")


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
        register_knowledge_tool(mcp, kit)
        return {t.name: t for t in asyncio.run(mcp.list_tools())}
    finally:
        queue.shutdown()


def test_every_knowledge_file_has_a_name_a_use_and_a_body():
    files = load_knowledge()
    assert set(files) == FILES
    for name, file in files.items():
        assert file["description"].startswith("Use when") and len(file["text"]) > 200, name
        assert "wbench" not in file["text"].lower()
    assert "2.4 s" in files["data_formats"]["text"] and "cam_c2w" in files["data_formats"]["text"]


def test_a_tool_file_exists_only_with_its_tool_and_starts_with_the_tools_description(tmp_path):
    from types import SimpleNamespace as NS
    schema = {"required": ["items"], "properties": {
        "items": {"description": "the clips", "anyOf": [{"type": "string"}, {"type": "array", "items": {
            "required": ["prompt"], "properties": {"prompt": {"description": "what it shows"},
                                                   "seed": {"description": "random seed", "enum": [1, 2]}}}}]},
        "frames": {"description": "8k+1"}}}
    tool = NS(description="What the tool says.", input_schema=schema)
    assert "caption_videos" not in load_knowledge() and "rollout_h3" not in load_knowledge({"caption_videos": tool})
    text = load_knowledge({"rollout_ltx25": tool})["rollout_ltx25"]["text"]
    assert text.startswith("# The tool\n\nWhat the tool says.\n\n# Its arguments\n\n"
                           "- `items`: the clips\n  - `prompt`: what it shows\n  - `seed` (optional): random seed (one of: 1, 2)\n"
                           "- `frames` (optional): 8k+1\n\n# What the published clip carries")
    tools = _tools(tmp_path)                                             # the mock kit has caption_videos only
    listed = tools["read_knowledge"].description
    assert "- caption_videos: Use when" in listed and "rollout_alayaworld" not in listed
    for name in ("rollout_alayaworld", "rollout_h3", "rollout_ltx25", "rollout_wan22", "generate_images",
                 "annotate_camera", "caption_videos", "locate_events"):
        text = load_knowledge({name: tool})[name]["text"]
        assert "wbench" not in text.lower() and "evaluation" not in text.lower()


def test_read_knowledge_lists_every_file_in_its_description(tmp_path):
    tool = _tools(tmp_path)["read_knowledge"]
    for name, file in load_knowledge().items():
        assert f"- {name}: {file['description']}" in tool.description


def test_every_kernel_tool_argument_is_described(tmp_path):
    tools = _tools(tmp_path)
    assert "filter" not in tools["data_query"].input_schema["properties"]
    assert set(tools["data_query"].input_schema["properties"]) == {"clip_ids", "ingested_by", "data_commit"}
    for name, tool in tools.items():
        for arg, schema in tool.input_schema["properties"].items():
            assert schema.get("description"), f"{name}.{arg} has no description"
    candidate = tools["data_ingest"].input_schema["properties"]["candidates"]["anyOf"][0]["items"]
    assert set(candidate["required"]) == {"video", "annotation", "pose", "derived_from"}
    assert all(p.get("description") for p in candidate["properties"].values())


def test_nothing_an_agent_reads_speaks_of_training_windows_rounds_or_removed_fields(tmp_path):
    """Knowledge files, and every description of the kernel's tools and of their arguments (the GPU tools with the
    real config)."""
    import json
    from ar_kernel.config import KernelConfig
    from ar_kernel.tools.gpu_jobs import build_gpu_backends, register_gpu_tools
    from ar_kernel.tools.locate_events import register_locate_tool
    rec = Recorder(tmp_path / "run")
    reg = TokenRegistry(rec)
    queue = JobQueue(rec, threading.Lock(), wait_cap_s=5.0)
    try:
        for backend in build_gpu_backends(KernelConfig.load(), tmp_path / "run", [0, 1, 2, 3], reg, rec):
            queue.register(backend)
        kit, mcp = ToolKit(reg, rec), new_mcp()
        register_gpu_tools(mcp, kit, queue)
        register_locate_tool(mcp, kit, queue)
        gpu_tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    finally:
        queue.shutdown()
    tools = {**_tools(tmp_path), **gpu_tools}
    texts = {f"knowledge {name}": file["text"] for name, file in load_knowledge(tools).items()}
    assert "knowledge rollout_h3" in texts and "knowledge locate_events" in texts
    for name, tool in tools.items():
        if name != "read_knowledge":                    # it lists the files, checked above
            texts[f"tool {name}"] = tool.description + json.dumps(tool.input_schema)
    assert "rollout_h3" in gpu_tools and "locate_events" in gpu_tools
    found = [(where, word) for where, text in texts.items() for word in AGENT_TEXT_NEVER_SAYS if word in text.lower()]
    assert found == []
