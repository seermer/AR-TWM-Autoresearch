"""The eval adapter's concatenation as a span."""
import subprocess

import pytest
from conftest import telemetry_records

from ar_kernel import telemetry
from ar_kernel.config import KernelConfig
from ar_kernel.eval import lora
from ar_kernel.telemetry import tree


def test_the_concatenation_is_a_span_ending_with_its_files_or_its_error(tmp_path, monkeypatch):
    telemetry.open(tmp_path / "run")
    telemetry.start("ar.run")

    def concat(env, args, cwd):
        out = args[args.index("--output") + 1]
        (tmp_path / "node" / "eval" / "lora").mkdir(parents=True, exist_ok=True)
        (tmp_path / "node" / "eval" / "lora" / "adapter.safetensors").write_bytes(b"x")
        return subprocess.CompletedProcess(args, 0 if out else 1, "", "")
    monkeypatch.setattr(lora, "run_in_env", concat)
    lora.concat_eval_lora(KernelConfig.load(), tmp_path / "ckpt", tmp_path / "node")
    monkeypatch.setattr(lora, "run_in_env", lambda env, args, cwd: subprocess.CompletedProcess(args, 1, "", "OOM"))
    with pytest.raises(RuntimeError, match="concatenation failed"):
        lora.concat_eval_lora(KernelConfig.load(), tmp_path / "ckpt", tmp_path / "node2")
    ends = [r for r in telemetry_records(tmp_path / "run") if r["kind"] == "end" and r["name"] == "eval.concat_lora"]
    assert [e["status"] for e in ends] == ["ok", "error"]
    assert tree.check(tree.load(tmp_path / "run")) == []
