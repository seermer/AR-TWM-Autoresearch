"""A stand-in for `vllm serve` in the caption_videos tests (no GPU, no model).

Called with the real command line minus `vllm serve`. `--fake-mode`: ok (default), exit (dies
during startup), hang (never becomes ready), slow (every caption takes 60 s). A clip whose bytes
start with BAD gets a 400, like a video the server cannot decode. Writes its pid to fake_vllm.pid
in its working directory.
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
if args.fake_mode == "exit":
    print("RuntimeError: CUDA out of memory while loading the model", flush=True)
    sys.exit(3)
if args.fake_mode == "hang":
    time.sleep(3600)
allowed = Path(args.allowed_local_media_path).resolve()


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
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        video, text = body["messages"][0]["content"]
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        path = Path(unquote(urlparse(video["video_url"]["url"]).path)).resolve()
        if allowed not in path.parents:
            return self.reply(400, {"error": {"message": f"{path} is outside the allowed media path"}})
        data = path.read_bytes()
        if data.startswith(b"BAD"):
            return self.reply(400, {"error": {"message": "failed to decode the video"}})
        if args.fake_mode == "slow":
            time.sleep(60)
        caption = f"{text['text']} [{len(data)} bytes, tp={args.tensor_parallel_size}]"
        self.reply(200, {"choices": [{"message": {"role": "assistant", "content": caption}}]})


ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
