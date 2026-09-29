import json

from scripts.make_proxy_cases import extend_proxy


def _case(types):
    return {"interactions": [{"type": t} for t in types]}


def test_extend_keeps_base_and_is_deterministic(tmp_path):
    kinds = [["navigation"], ["event_edit"], ["subject_action"], ["perspective_switch"]]
    for i in range(1, 41):
        (tmp_path / f"case_{i}.json").write_text(json.dumps(_case(kinds[i % 4])))
    base = [str(i) for i in range(1, 9)]
    a = extend_proxy(tmp_path, base, 12, seed=1)
    assert a == extend_proxy(tmp_path, base, 12, seed=1)
    assert set(base) <= set(a) and len(a) == 12 and len(set(a)) == 12
    assert a == sorted(a, key=int)


def test_extend_balances_interaction_mix(tmp_path):
    for i in range(1, 41):
        (tmp_path / f"case_{i}.json").write_text(json.dumps(_case(["navigation"] if i <= 20 else ["event_edit"])))
    base = [str(i) for i in range(1, 9)]  # all navigation: the extension must add event_edit
    out = extend_proxy(tmp_path, base, 12, seed=1)
    assert all(int(i) > 20 for i in out if i not in base)
