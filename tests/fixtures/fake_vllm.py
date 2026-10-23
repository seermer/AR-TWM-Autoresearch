"""A stand-in for `vllm serve` in the caption_videos tests (no GPU, no model).

Called with the real command line minus `vllm serve`. `--fake-mode`: ok (default), exit (dies
during startup), hang (never becomes ready), slow (every caption takes 60 s), second (1 s each). A clip whose bytes
start with BAD gets a 400, like a video the server cannot decode. Writes its pid to fake_vllm.pid
and its argv to fake_vllm.argv.json in its working directory; refuses to start without
HF_HUB_OFFLINE=1 or VLLM_API_KEY; captions name the CUDA_VISIBLE_DEVICES it was given. A request
without the key gets a 401, as from vLLM.

A request with a `response_format` is a locate_events frame: the reply is JSON in the asked shape.
An event whose sentence holds NEVER is not there; any other is happening. A request whose event holds
BROKEN gets text that is not JSON.
"""
import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--host")
parser.add_argument("--port", type=int)
parser.add_argument("--allowed-local-media-path")
parser.add_argument("--tensor-parallel-size")
parser.add_argument("--fake-mode", default="ok")
args, _ = parser.parse_known_args()
Path("fake_vllm.pid").write_text(str(os.getpid()))
Path("fake_vllm.argv.json").write_text(json.dumps(sys.argv[1:]))
if os.environ.get("HF_HUB_OFFLINE") != "1":
    print("HF_HUB_OFFLINE is not 1: the server could download", flush=True)
    sys.exit(4)
if not os.environ.get("VLLM_API_KEY"):
    print("no VLLM_API_KEY: anyone on the machine could call the server", flush=True)
    sys.exit(5)
if args.fake_mode == "exit":
    print("RuntimeError: CUDA out of memory while loading the model", flush=True)
    sys.exit(3)
if args.fake_mode == "hang":
    time.sleep(3600)
allowed = Path(args.allowed_local_media_path).resolve() if args.allowed_local_media_path else None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.reply(200 if self.path == "/health" else 404, {})

    def do_POST(self):
        if self.headers.get("Authorization") != f"Bearer {os.environ['VLLM_API_KEY']}":
            return self.reply(401, {"error": "Unauthorized"})
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if "response_format" in body:
            return self.locate(body)
        system, user = body["messages"]
        assert system["role"] == "system" and "video reader" in system["content"]
        video, text = user["content"]
        assert body["reasoning_effort"] == "low" and "temperature" not in body and "max_tokens" not in body and "chat_template_kwargs" not in body
        path = Path(unquote(urlparse(video["video_url"]["url"]).path)).resolve()
        if allowed not in path.parents:
            return self.reply(400, {"error": {"message": f"{path} is outside the allowed media path"}})
        data = path.read_bytes()
        if data.startswith(b"BAD"):
            return self.reply(400, {"error": {"message": "failed to decode the video"}})
        if args.fake_mode == "slow":
            time.sleep(60)
        if args.fake_mode == "second":        # every caption takes 1 s, requests are served in parallel
            time.sleep(1)
        caption = (f"{text['text']} [{len(data)} bytes, tp={args.tensor_parallel_size}, "
                   f"gpus={os.environ['CUDA_VISIBLE_DEVICES']}]")
        self.reply(200, {"choices": [{"message": {"role": "assistant", "content": caption,
                                                  "reasoning_content": f"thinking about {len(data)} bytes"}}]})


    def locate(self, body):
        system, user = body["messages"]
        assert system["role"] == "system" and "frame reader" in system["content"]
        image, text = user["content"]
        assert body["reasoning_effort"] == "low" and "temperature" not in body and image["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert set(body["response_format"]["json_schema"]["schema"]["properties"]) == {"happening_now", "already_happened"}
        event = text["text"].removeprefix("Event: ")
        answer = {"happening_now": "NEVER" not in event, "already_happened": False}
        content = "I cannot tell." if "BROKEN" in event else json.dumps(answer)
        self.reply(200, {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                         "choices": [{"index": 0, "finish_reason": "stop",
                                      "message": {"role": "assistant", "content": content,
                                                  "reasoning_content": "looked at one frame"}}]})


ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
