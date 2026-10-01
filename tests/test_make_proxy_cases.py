import json
from collections import Counter

from scripts.make_proxy_cases import TYPES, order_cases


def _write(tmp_path, cases):
    for cid, (types, causal) in cases.items():
        case = {"interactions": [{"type": t} for t in types]}
        if causal:
            case["causal_fidelity"] = {}
        (tmp_path / f"case_{cid}.json").write_text(json.dumps(case))


def test_order_covers_every_case_and_is_deterministic(tmp_path):
    _write(tmp_path, {str(i): ([TYPES[i % 4]], False) for i in range(1, 41)})
    a = order_cases(tmp_path, seed=1)
    assert a == order_cases(tmp_path, seed=1)
    assert sorted(a, key=int) == [str(i) for i in range(1, 41)]


def test_every_prefix_keeps_the_types_roughly_equal(tmp_path):
    cases = {str(i): (["navigation"], False) for i in range(1, 31)}
    cases |= {str(i): ([TYPES[1 + i % 3]], False) for i in range(31, 61)}
    _write(tmp_path, cases)
    order = order_cases(tmp_path, seed=1)
    for n in (4, 13, 40):
        counts = Counter(cases[i][0][0] for i in order[:n])
        assert max(counts.values()) - min(counts[t] for t in TYPES) <= 1


def test_navigation_takes_causal_cases_first_and_single_type_cases_lead(tmp_path):
    cases = {str(i): (["navigation"], i > 6) for i in range(1, 9)}                  # 7, 8 are causal
    cases |= {"9": (["event_edit", "subject_action"], False), "10": (["event_edit"], False),
              "11": (["subject_action"], False), "12": (["perspective_switch"], False)}
    _write(tmp_path, cases)
    order = order_cases(tmp_path, seed=3)
    navigation = [i for i in order if cases[i][0] == ["navigation"]]
    assert set(navigation[:2]) == {"7", "8"}
    assert order.index("10") < order.index("9") and order.index("11") < order.index("9")
