"""Extend the proxy case list to a target size, keeping the interaction-type mix close to all WBench cases.

    python scripts/make_proxy_cases.py --target 50 --seed 20260928 --write
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT.parent / "WBench" / "data" / "cases"
PROXY = ROOT / "configs" / "proxy_cases.txt"


def _types(cases_dir: Path) -> dict[str, frozenset]:
    out = {}
    for p in cases_dir.glob("case_*.json"):
        cid = p.stem.removeprefix("case_")
        out[cid] = frozenset(i["type"] for i in json.loads(p.read_text())["interactions"])
    return out


def _mix(ids, types) -> dict[str, float]:
    c = Counter(t for i in ids for t in types[i])
    total = sum(c.values()) or 1
    return {t: n / total for t, n in c.items()}


def _distance(a: dict, b: dict) -> float:
    return sum(abs(a.get(t, 0.0) - b.get(t, 0.0)) for t in set(a) | set(b))


def extend_proxy(cases_dir: Path, base_ids: list[str], target: int, seed: int) -> list[str]:
    """`base_ids` plus greedily chosen cases so the interaction-type mix matches the whole set."""
    types = _types(cases_dir)
    goal = _mix(list(types), types)
    chosen = list(base_ids)
    rest = [i for i in types if i not in chosen]
    random.Random(seed).shuffle(rest)  # shuffled order breaks distance ties
    while len(chosen) < target and rest:
        best = min(rest, key=lambda i: _distance(_mix(chosen + [i], types), goal))
        chosen.append(best)
        rest.remove(best)
    return sorted(chosen, key=int)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    base = PROXY.read_text().strip().split(",")
    out = extend_proxy(CASES, base, a.target, a.seed)
    print(",".join(out))
    if a.write:
        PROXY.write_text(",".join(out) + "\n")


if __name__ == "__main__":
    main()
