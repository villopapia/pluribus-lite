"""M4 (D-SRCH-3): the subgame roots Pluribus actually searched from, for calibrating the search
action abstraction.

Search is always used on rounds 2-4 and its subgame starts at the beginning of the current round
(Supp pp. 19-21). For every hand and every flop/turn/river start where Pluribus is still in the
hand and at least two players can still act, write the round-start state: street, folded mask,
and each seat's total contribution. Seats p1..p6 = our seats 0..5 (p1 = small blind).

Output: data/subgame_roots.tsv with columns
  hand_id  street  folded_mask  t0 .. t5        (street: 1 flop, 2 turn, 3 river)

Usage:  .venv/Scripts/python subgame_roots.py [--data-dir ../data]
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from m0_parse import hand_files

STACK = 10_000


def roots_of(actions: list[str], bot_seat: int, n: int = 6) -> list[tuple[int, int, list[int]]]:
    """(street, folded_mask, totals) at each post-flop round start worth searching."""
    total = [50, 100] + [0] * (n - 2)
    street_bet = list(total)
    folded = 0
    street = 0
    out = []
    for a in actions:
        p = a.split()
        if p[0] == "d":
            if p[1] == "db":
                street += 1
                street_bet = [0] * n
                alive = [s for s in range(n) if not folded >> s & 1]
                can_act = [s for s in alive if total[s] < STACK]
                if not folded >> bot_seat & 1 and len(can_act) >= 2:
                    out.append((street, folded, list(total)))
            continue
        s = int(p[0][1:]) - 1
        if p[1] == "f":
            folded |= 1 << s
        elif p[1] == "cc":
            amt = min(max(street_bet) - street_bet[s], STACK - total[s])
            street_bet[s] += amt
            total[s] += amt
        elif p[1] == "cbr":
            to = int(p[2])
            total[s] += to - street_bet[s]
            street_bet[s] = to
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--data-dir", type=Path, default=here.parent / "data")
    args = ap.parse_args(argv)
    files = hand_files(args.data_dir / "phh-dataset" / "data" / "pluribus")
    rows = []
    for f in files:
        h = tomllib.loads(f.read_text(encoding="utf-8"))
        seat = h["players"].index("Pluribus")
        for street, folded, total in roots_of(h["actions"], seat):
            rows.append(f"{f.parent.name}/{f.stem}\t{street}\t{folded}\t" + "\t".join(map(str, total)))
    out = args.data_dir / "subgame_roots.tsv"
    out.write_text("hand_id\tstreet\tfolded_mask\tt0\tt1\tt2\tt3\tt4\tt5\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"{len(files)} hands -> {len(rows)} post-flop subgame roots in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
