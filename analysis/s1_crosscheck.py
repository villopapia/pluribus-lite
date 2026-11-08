"""D-DATA-1: hand-by-hand cross-check of the PHH conversion against the original logs
(Science Data File S1, `aay2400_data_file_s1.zip`, unzipped to data/s1/5H1AI_logs/).

S1 format (README.txt in the zip; ACPC "STATE" lines), one hand per line:

    STATE:<index>:<actions>:<hole|hole|...>/<flop>/<turn>/<river>:<payoffs>:<players>

* actions: rounds separated by '/', 'f' fold, 'c' check/call, 'r<N>' raise to N = the
  raiser's **total chips in the pot for the whole hand** (not per street);
* all six hole-card pairs are listed (folded players too), seat order = players order,
  first player = small blind;
* payoffs: chips won/lost per player.

PHH (`phh-dataset/data/pluribus/<game>/<index>.phh`) uses per-street raise-to amounts
('cbr N' = the player's street bet after the raise), so S1 action strings are rebuilt from
PHH and compared exactly. Compared per hand: players, hole cards (all 6), board, action
string, payoffs (PHH finishing - starting stacks).

Usage:  .venv/Scripts/python s1_crosscheck.py [--s1 ../data/s1/5H1AI_logs]
          [--phh ../data/phh-dataset/data/pluribus] [--out ../docs/s1_crosscheck.md]
"""

from __future__ import annotations

import argparse
import re
import tomllib
from collections import Counter
from pathlib import Path


def parse_s1_line(line: str) -> dict:
    _, idx, actions, cards, payoffs, players = line.strip().split(":")
    parts = cards.split("/")
    return {
        "index": int(idx),
        "actions": actions,
        "holes": parts[0].split("|"),
        "board": "".join(parts[1:]),
        "payoffs": [float(x) for x in payoffs.split("|")],
        "players": players.split("|"),
    }


def phh_to_acpc(h: dict) -> dict:
    """Rebuild the S1 fields from one parsed PHH hand."""
    n = len(h["players"])
    stacks = list(h["starting_stacks"])
    blinds = list(h["blinds_or_straddles"])
    street_bet = blinds[:n]
    total = blinds[:n]
    holes = [""] * n
    board = ""
    rounds: list[str] = [""]
    for a in h["actions"]:
        t = a.split()
        if t[0] == "d" and t[1] == "dh":
            holes[int(t[2][1:]) - 1] = t[3]
        elif t[0] == "d" and t[1] == "db":
            board += t[2]
            rounds.append("")
            street_bet = [0] * n
        elif t[0].startswith("p"):
            p = int(t[0][1:]) - 1
            if t[1] == "f":
                rounds[-1] += "f"
            elif t[1] == "cc":
                to = min(max(street_bet), street_bet[p] + stacks[p] - total[p])
                total[p] += to - street_bet[p]
                street_bet[p] = to
                rounds[-1] += "c"
            elif t[1] == "cbr":
                to = int(t[2])
                total[p] += to - street_bet[p]
                street_bet[p] = to
                rounds[-1] += f"r{total[p]}"
            # 'sm' (show/muck) carries no action
    payoffs = [float(f - s) for f, s in zip(h["finishing_stacks"], h["starting_stacks"])]
    return {"actions": "/".join(rounds), "holes": holes, "board": board, "payoffs": payoffs, "players": list(h["players"])}


def norm_actions(s: str) -> str:
    return s.rstrip("/")


def compare(s1: dict, ours: dict) -> list[str]:
    diffs = []
    for k in ["players", "holes", "board"]:
        if s1[k] != ours[k]:
            diffs.append(k)
    if norm_actions(s1["actions"]) != norm_actions(ours["actions"]):
        diffs.append("actions")
    if any(abs(a - b) > 1e-9 for a, b in zip(s1["payoffs"], ours["payoffs"])):
        diffs.append("payoffs")
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", default="../data/s1/5H1AI_logs")
    ap.add_argument("--phh", default="../data/phh-dataset/data/pluribus")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    s1_dir, phh_dir = Path(a.s1), Path(a.phh)
    n = 0
    missing: list[str] = []
    kinds: Counter = Counter()
    examples: dict[str, list[str]] = {}
    for log in sorted(s1_dir.glob("sample_game_*.log")):
        game = re.match(r"sample_game_(\w+)\.log", log.name).group(1)
        for line in log.read_text().splitlines():
            if not line.startswith("STATE:"):
                continue
            n += 1
            s = parse_s1_line(line)
            f = phh_dir / game / f"{s['index']}.phh"
            if not f.exists():
                missing.append(f"{game}/{s['index']}")
                continue
            ours = phh_to_acpc(tomllib.loads(f.read_text()))
            d = compare(s, ours)
            for k in d:
                kinds[k] += 1
                ex = examples.setdefault(k, [])
                if len(ex) < 5:
                    ex.append(f"{game}/{s['index']}: S1 {s[k]} | PHH {ours[k]}")
            kinds["_any"] += bool(d)
    phh_total = sum(1 for _ in phh_dir.glob("*/*.phh"))
    lines = [
        f"- S1 hands: {n}; PHH hands: {phh_total}; S1 hands without a PHH file: {len(missing)}",
        f"- hands identical in all fields: {n - len(missing) - kinds['_any']}",
        f"- hands with any mismatch: {kinds['_any']}",
    ]
    for k in ["players", "holes", "board", "actions", "payoffs"]:
        lines.append(f"  - {k}: {kinds[k]} mismatching hands")
        for e in examples.get(k, []):
            lines.append(f"    - `{e}`")
    if missing:
        lines.append(f"- missing PHH files (first 5): {missing[:5]}")
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
