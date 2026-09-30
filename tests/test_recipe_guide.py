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
    assert all(set(g[k]) == {"base", "meaning"} and g[k]["meaning"] for k in g)


def _line(i, loss, sigma=0.930, grad=0.1):
    return (f"[Train] step={i} epoch=0 source=s video=abc fs=1 fe=2 K=4 gap=0 cond=i2v:1 control=[] spatial=4 "
            f"mem_drop=1 hist=drift prefix_fix=0 sigma={sigma} loss={loss} grad={grad} lr=2.00e-05 time=14.0s\n")


def test_train_summary_reads_loss_per_sigma_bin_and_half(tmp_path):
    log = tmp_path / "train.log"
    lines = [_line(i, (1.0 if i <= 50 else 0.5) if i % 2 else 2.0, sigma=0.9 if i % 2 else 0.1) for i in range(1, 101)]
    log.write_text("starting\n" + "".join(lines))
    s = train_summary(log)
    assert s["steps"] == 100 and s["mean_grad_norm"] == 0.1
    assert s["loss_by_sigma"]["sigma>=0.6"] == {"first_half": 1.0, "second_half": 0.5}
    assert s["loss_by_sigma"]["sigma<0.3"] == {"first_half": 2.0, "second_half": 2.0}
    assert s["loss_by_sigma"]["0.3<=sigma<0.6"] == {"first_half": None, "second_half": None}


def test_train_summary_missing_or_empty_log_is_empty(tmp_path):
    assert train_summary(tmp_path / "nope.log") == {}
    (tmp_path / "e.log").write_text("no steps here\n")
    assert train_summary(tmp_path / "e.log") == {}


def test_train_summary_reads_the_real_acceptance_log_format():
    from pathlib import Path
    log = Path(__file__).resolve().parents[1] / "runs/live-09-29/nodes/n1/attempts/improve_recipe-1/train/train.log"
    if not log.exists():
        return
    s = train_summary(log)
    assert s["steps"] > 50 and s["loss_by_sigma"]["sigma>=0.6"]["first_half"] > 0
