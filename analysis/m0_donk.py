"""M0 follow-up: donk-bet audit -> docs/m0_donk.md

Donk-bet definition (strict, per street s = flop/turn/river):
  Opportunity for player X on street s when
    1. street s-1 had at least one voluntary bet/raise ('cbr'); let A = the last bettor/raiser,
    2. X != A and X *called* A's last bet on street s-1 (X's last action there is 'cc' after it),
    3. X is the first player to act on street s.
  Donk bet = X bets ('cbr') at that opportunity.

Computed for every player, so Pluribus can be compared with the humans in the same hands.
Also recomputes the looser definition used in docs/m0_stats.md (v1) for comparison.

Usage:  .venv/Scripts/python -I m0_donk.py
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
from pokerkit import HandHistory

from m0_parse import POSITIONS, hand_files

STREETS = ["preflop", "flop", "turn", "river"]
BOT = "Pluribus"


def split_streets(actions: list[str]) -> tuple[list[list[tuple[int, str, int | None]]], list[str]]:
    """Player actions per street as (seat0, verb, amount); also boards dealt."""
    streets: list[list[tuple[int, str, int | None]]] = [[]]
    boards: list[str] = []
    for a in actions:
        p = a.split()
        if p[0] == "d":
            if p[1] == "db":
                streets.append([])
                boards.append(p[2])
            continue
        if p[1] == "sm":
            continue
        amt = int(p[2]) if p[1] == "cbr" else None
        streets[-1].append((int(p[0][1:]) - 1, p[1], amt))
    return streets, boards


def donk_spots(actions: list[str], players: list[str], hand_id: str) -> list[dict]:
    streets, boards = split_streets(actions)
    out = []
    for s in range(1, len(streets)):
        cur, prev = streets[s], streets[s - 1]
        if not cur:
            continue
        cbr_idx = [i for i, (_, v, _) in enumerate(prev) if v == "cbr"]
        if not cbr_idx:
            continue
        last = cbr_idx[-1]
        aggressor = prev[last][0]
        x, verb, amt = cur[0]
        if x == aggressor:
            continue
        after = [v for (seat, v, _) in prev[last + 1 :] if seat == x]
        if not after or after[-1] != "cc":
            continue
        # players who put in the full last bet: the aggressor plus everyone whose last action
        # after it was a call
        last_act: dict[int, str] = {}
        for seat, v, _ in prev[last + 1 :]:
            last_act[seat] = v
        n_in = 1 + sum(1 for v in last_act.values() if v == "cc")
        out.append(
            {
                "hand_id": hand_id,
                "player": players[x],
                "is_bot": players[x] == BOT,
                "position": POSITIONS[x],
                "street": STREETS[s],
                "aggressor": players[aggressor],
                "aggressor_position": POSITIONS[aggressor],
                "players_seeing_street": n_in,
                "board": "".join(boards[:s]),
                "donk": verb == "cbr",
                "amount_to": amt,
                "prev_street_line": " ".join(
                    f"{POSITIONS[seat]}:{v}{'' if a is None else ' ' + str(a)}" for seat, v, a in prev
                ),
            }
        )
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["street", "is_bot"]).donk.agg(opps="size", donks="sum").reset_index()
    g["rate"] = g.donks / g.opps
    return g


def main() -> int:
    here = Path(__file__).resolve().parent
    root = here.parent / "data" / "phh-dataset" / "data" / "pluribus"
    rows = []
    for f in hand_files(root):
        with f.open("rb") as fh:
            hh = HandHistory.load(fh)
        rows += donk_spots(list(hh.actions), list(hh.players), f"{f.parent.name}/{f.stem}")
    df = pd.DataFrame(rows)

    d = pd.read_parquet(here.parent / "data" / "pluribus_decisions.parquet")
    flop_first = d[d.street == "flop"].groupby("hand_id").head(1)
    v1 = flop_first[(flop_first.raises_before == 0) & ~flop_first.preflop_aggressor & flop_first.prev_aggressor_acts_after]
    v1_ids = set(v1.hand_id)
    strict_flop = df[(df.street == "flop") & df.is_bot]
    strict_ids = set(strict_flop.hand_id)

    L = ["# M0: donk-bet audit", "",
         "Definition (strict): on street s, player X **called the last bet/raise** of street s-1 "
         "(made by someone else), X is **first to act** on street s, and X **bets**. "
         "Denominator = all such spots. Computed for every player in the 10,000 hands.", "",
         "## Rates by street", "",
         "| street | Pluribus opps | Pluribus donks | Pluribus rate | humans opps | humans donks | humans rate |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    sm = summarize(df)
    for s in STREETS[1:]:
        b = sm[(sm.street == s) & sm.is_bot].iloc[0]
        h = sm[(sm.street == s) & ~sm.is_bot].iloc[0]
        L.append(f"| {s} | {b.opps:,} | {b.donks:,} | {100 * b.rate:.1f}% | {h.opps:,} | {h.donks:,} | {100 * h.rate:.1f}% |")
    tb, th = df[df.is_bot], df[~df.is_bot]
    L.append(f"| **all** | {len(tb):,} | {tb.donk.sum():,} | {100 * tb.donk.mean():.1f}% | "
             f"{len(th):,} | {th.donk.sum():,} | {100 * th.donk.mean():.1f}% |")
    L.append("")

    L += ["## Heads-up vs multiway (flop, Pluribus)", "",
          "| players who called into the flop (incl. aggressor) | opps | donk rate |", "|---|---:|---:|"]
    for n, g in strict_flop.groupby("players_seeing_street"):
        L.append(f"| {n} | {len(g)} | {100 * g.donk.mean():.1f}% |")
    L.append("")

    L += ["## Why the earlier figure was 2.1% (v1 in m0_stats.md)", "",
          f"v1 flop spots: {len(v1)} (donks {int((v1.action == 'bet').sum())}, "
          f"{100 * (v1.action == 'bet').mean():.1f}%). Strict flop spots: {len(strict_flop)} "
          f"(donks {int(strict_flop.donk.sum())}, {100 * strict_flop.donk.mean():.1f}%).", "",
          f"- In v1 but not strict: {len(v1_ids - strict_ids)} hands",
          f"- In strict but not v1: {len(strict_ids - v1_ids)} hands",
          f"- In both: {len(v1_ids & strict_ids)} hands", ""]
    only_v1 = v1[~v1.hand_id.isin(strict_ids)]
    if len(only_v1):
        # v1 only required Pluribus to act before the preflop aggressor; strict also requires it
        # to be first to act. Count v1-only spots where someone checked before Pluribus.
        checked_before = only_v1.history.map(lambda h: h.rsplit("|", 1)[-1].strip(",") != "").sum()
        L += [f"Of the {len(only_v1)} v1-only spots, {checked_before} had another player check "
              "before Pluribus on the flop (so Pluribus was not first to act). v1 also counted only "
              "the flop. Under both definitions Pluribus' flop donk rate is low; its donking is "
              "concentrated on the turn.", ""]

    rng = random.Random(0)
    cols = ["hand_id", "street", "position", "aggressor_position", "board", "prev_street_line", "amount_to"]
    for label, sub in [("donk bets", tb[tb.donk]), ("checks (no donk)", tb[~tb.donk])]:
        idx = sorted(rng.sample(list(sub.index), min(10, len(sub))))
        L += [f"## 10 sample Pluribus spots: {label}", "",
              "| hand | street | Pluribus | aggressor | board | previous-street line | bet to |", "|---|---|---|---|---|---|---:|"]
        for _, r in sub.loc[idx, cols].iterrows():
            amt = "" if pd.isna(r.amount_to) else int(r.amount_to)
            L.append(f"| {r.hand_id} | {r.street} | {r.position} | {r.aggressor_position} | {r.board} | {r.prev_street_line} | {amt} |")
        L.append("")

    out = here.parent / "docs" / "m0_donk.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
