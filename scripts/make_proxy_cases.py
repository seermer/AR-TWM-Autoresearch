"""Order all WBench cases so that every prefix is a usable proxy set (a run scores the first
`eval.proxy_size` of them).

    python scripts/make_proxy_cases.py --seed 20261001 --write

Each next case is of the interaction type the list holds fewest of, so the four types stay
roughly equal in every prefix until a type runs out. Within a type, cases with only that type
come first, and navigation takes cases graded on causal fidelity first.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT.parent / "WBench" / "data" / "cases"
PROXY = ROOT / "configs" / "proxy_cases.txt"
TYPES = ("navigation", "event_edit", "subject_action", "perspective_switch")


def order_cases(cases_dir: Path, seed: int) -> list[str]:
    types, causal = {}, set()
    for p in sorted(cases_dir.glob("case_*.json")):
        cid = p.stem.removeprefix("case_")
        case = json.loads(p.read_text())
        types[cid] = frozenset(i["type"] for i in case["interactions"])
        if "causal_fidelity" in case:
            causal.add(cid)
    rest = sorted(types)
    random.Random(seed).shuffle(rest)          # the shuffled order breaks ties
    ordered, counts = [], Counter()
    while rest:
        wanted = min((t for t in TYPES if any(t in types[i] for i in rest)), key=lambda t: counts[t])
        pick = min((i for i in rest if wanted in types[i]),
                   key=lambda i: (wanted != "navigation" or i not in causal, len(types[i])))
        ordered.append(pick)
        rest.remove(pick)
        counts.update(types[pick])
    return ordered


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    out = order_cases(CASES, a.seed)
    print(",".join(out))
    if a.write:
        PROXY.write_text(",".join(out) + "\n")


if __name__ == "__main__":
    main()
