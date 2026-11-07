"""caption_videos bridge (runs in the vllm env): starts `vllm serve`, captions every clip through its
OpenAI-compatible endpoint, and stops it.

python caption_vllm.py --job <work>/job.json (bridges/protocol.py). Job file: serve ({"argv", "env", "port"}: the
command the kernel built), system, prompt, concurrent_requests, clip_timeout_s, reasoning_effort; items {index,
path}. One item message per clip with its caption and reasoning, or its error; requests run in threads, so every
item's log is the server's.
"""
from __future__ import annotations

import os
import secrets
import subprocess
import sys
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import protocol


def serve(job: dict) -> tuple[subprocess.Popen, str, str, float]:
    """`job["serve"]` started with its own API key (the sandbox shares the host network), output to logs/vllm.log;
    (process, base URL, key, load seconds) once /health answers. A server that exits first ends the bridge with
    its log's tail."""
    s, key, log = job["serve"], secrets.token_urlsafe(24), protocol.log_path(job, "vllm")
    t0, url = time.monotonic(), f"http://127.0.0.1:{s['port']}"
    with open(log, "w", encoding="utf-8") as sink:
        proc = subprocess.Popen(s["argv"], cwd=job["work"], env={**os.environ, **s["env"], "VLLM_API_KEY": key},
                                stdout=sink, stderr=subprocess.STDOUT)
    http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while True:
        if proc.poll() is not None:
            sys.exit(f"the vLLM server exited during startup (exit {proc.returncode}):\n"
                     f"{log.read_text(errors='replace')[-3000:]}")
        try:
            with http.open(f"{url}/health", timeout=5) as r:
                if r.status == 200:
                    return proc, url, key, time.monotonic() - t0
        except OSError:
            pass
        time.sleep(2)


def stop(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def client(url: str, key: str, timeout_s: float):
    """The OpenAI client for the server on this machine: no proxy lookup."""
    import httpx
    import openai
    return openai.OpenAI(base_url=f"{url}/v1", api_key=key, http_client=httpx.Client(timeout=timeout_s, trust_env=False))


def caption(client, file: Path, system: str, prompt: str, reasoning_effort: str) -> dict:
    """{"caption"} or {"error"}, with "reasoning" when the model gave it."""
    import openai
    # No max_tokens: the model reasons first, and a cap could cut the caption off.
    try:
        done = client.chat.completions.create(
            model="captioner", reasoning_effort=reasoning_effort,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": [{"type": "video_url", "video_url": {"url": Path(file).as_uri()}},
                                                   {"type": "text", "text": prompt}]}])
    except openai.OpenAIError as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:2000]}
    message = done.choices[0].message
    reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
    text = (message.content or "").strip()
    out = {"caption": text} if text else {"error": "empty caption"}
    return {**out, "reasoning": reasoning} if reasoning else out


def main() -> int:
    job = protocol.load_job()
    messages = protocol.messages(job)
    proc, url, key, load_s = serve(job)
    messages.send("ready", info={"load_s": round(load_s, 2)})
    log = str(protocol.log_path(job, "vllm"))

    def one(c, item) -> None:
        index, started = item["index"], time.time()
        messages.send("item_started", index=index, log=log)
        try:
            got = caption(c, item["path"], job["system"], job["prompt"], job["reasoning_effort"])
        except Exception as exc:          # noqa: BLE001 -- this clip's error; the others go on
            messages.send("item", index=index, ok=False, error=f"{type(exc).__name__}: {exc}"[:2000],
                          traceback=traceback.format_exc(), started=started, ended=time.time(), log=log)
            return
        fields = ({"ok": True, "outputs": {}, "info": got} if "caption" in got else
                  {"ok": False, "error": got["error"], "info": got})
        messages.send("item", index=index, **fields, started=started, ended=time.time(), log=log)

    try:
        with client(url, key, float(job["clip_timeout_s"])) as c, \
                ThreadPoolExecutor(max_workers=max(1, min(int(job["concurrent_requests"]), len(job["items"])))) as pool:
            for f in [pool.submit(one, c, item) for item in job["items"]]:
                f.result()
    finally:
        stop(proc)
    messages.send("job_done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
