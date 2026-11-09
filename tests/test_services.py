"""The run's services as one span, from start to stop: their socket changes and the servers' own log lines."""
import logging

import httpx
from conftest import client_dir, telemetry_payload, telemetry_records
from fastapi import FastAPI

from ar_kernel import telemetry
from ar_kernel.services import RunServices, socket_dir_for
from ar_kernel.telemetry import capture


def test_services_are_one_span_holding_their_socket_changes_and_log_lines(tmp_path):
    run = tmp_path / "run"
    telemetry.open(run)
    capture.install()
    telemetry.start("ar.run")
    app = FastAPI()

    @app.get("/x")
    async def x():
        logging.getLogger("uvicorn.error").warning("from the server thread")
        return {}
    services = RunServices(socket_dir_for(run))
    services.start(app, FastAPI())
    try:
        with httpx.Client(transport=httpx.HTTPTransport(uds=str(client_dir(services) / "gateway.sock"))) as client:
            assert client.get("http://ar/x").status_code == 200
    finally:
        services.stop()
    records = telemetry_records(run)
    [span] = [r for r in records if r["kind"] == "start" and r["name"] == "services"]
    children = [r["name"] for r in records if r["kind"] == "start" and r["parent"] == span["span"]]
    assert children == ["fs.mkdir", "fs.chmod", "fs.chmod", "fs.chmod", "fs.remove", "fs.remove"]
    [line] = [r for r in records if r["kind"] == "log" and r["text"] == "from the server thread"]
    [end] = [r for r in records if r["kind"] == "end" and r["span"] == span["span"]]
    assert line["span"] == span["span"] and line["logger"] == "uvicorn.error"
    assert end["status"] == "ok" and telemetry_payload(run, end["output"]) == {"stuck": []}
