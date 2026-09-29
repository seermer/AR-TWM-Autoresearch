from ar_kernel.train.recipe import TUNABLE_KEYS
from ar_kernel.train.recipe_guide import recipe_guide, train_summary

BASE = {"optimizer": {"lr": 5e-5, "max_steps": 300, "grad_accum_steps": 4, "warmup_steps": 50,
                      "weight_decay": 0.001, "max_grad_norm": 5.0, "epochs": 5000},
        "lora": {"rank": 64, "alpha": 64}, "sample": {"height": 416, "width": 736}}


def test_guide_covers_every_tunable_key_with_base_values():
    g = recipe_guide(BASE)
    assert set(g) == set(TUNABLE_KEYS)
    assert g["optimizer.lr"]["base"] == 5e-5 and g["lora.rank"]["base"] == 64
    assert g["data.overall_caption_prob"]["base"] is None            # absent from the base recipe
    assert all(g[k]["meaning"] and g[k]["try"] and g[k]["cost"] for k in g)


def _line(i, loss, grad=0.1, t=14.0):
    return (f"[Train] step={i} epoch=0 source=s video=abc fs=1 fe=2 K=4 gap=0 cond=i2v:1 control=[] spatial=4 "
            f"mem_drop=1 hist=drift prefix_fix=0 sigma=0.930 loss={loss} grad={grad} lr=2.00e-05 time={t}s\n")


def test_train_summary_reads_loss_quarters(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("starting\n" + "".join(_line(i, 1.0 if i <= 25 else 0.5) for i in range(1, 101)))
    s = train_summary(log)
    assert s["steps"] == 100 and s["loss_first_quarter"] == 1.0 and s["loss_last_quarter"] == 0.5
    assert s["mean_grad_norm"] == 0.1 and s["sec_per_step"] == 14.0


def test_train_summary_missing_or_empty_log_is_empty(tmp_path):
    assert train_summary(tmp_path / "nope.log") == {}
    (tmp_path / "e.log").write_text("no steps here\n")
    assert train_summary(tmp_path / "e.log") == {}


def test_train_summary_reads_the_real_acceptance_log_format():
    from pathlib import Path
    log = Path(__file__).resolve().parents[1] / "runs/acceptance_20260928/nodes/n1/attempts/improve_recipe-1/train/train.log"
    if not log.exists():
        return
    s = train_summary(log)
    assert s["steps"] > 50 and s["sec_per_step"] > 0
