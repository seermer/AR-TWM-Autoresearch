from ar_kernel.train.recipe import TUNABLE_KEYS
from ar_kernel.train.recipe_guide import recipe_guide

BASE = {"optimizer": {"lr": 5e-5, "max_steps": 300, "grad_accum_steps": 4, "warmup_steps": 50,
                      "weight_decay": 0.001, "max_grad_norm": 5.0, "epochs": 5000},
        "lora": {"rank": 64, "alpha": 64}, "sample": {"height": 416, "width": 736}}


def test_guide_covers_every_tunable_key_with_base_values():
    g = recipe_guide(BASE)
    assert set(g) == set(TUNABLE_KEYS)
    assert g["optimizer.lr"]["base"] == 5e-5 and g["lora.rank"]["base"] == 64
    assert g["data.overall_caption_prob"]["base"] is None            # absent from the base recipe
    assert all(set(g[k]) == {"base", "meaning"} and g[k]["meaning"] for k in g)
