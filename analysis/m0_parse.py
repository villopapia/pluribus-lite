"""M0: parse the Pluribus hands from uoftcprg/phh-dataset into parquet.

Outputs (both in data/):
  pluribus_decisions.parquet  one row per Pluribus decision (fold / check / call / bet / raise)
  pluribus_hands.parquet      one row per hand (Pluribus seat, net chips, hole-card completeness)

Usage:  .venv/Scripts/python -I m0_parse.py [--data-dir ../data]

Definitions (all chip amounts in chips; big blind = 100 chips):
  pot            total chips in the middle *before* the decision, including bets on the current street
  to_call        chips Pluribus must add to call (0 = may check), capped at its stack
  stack          Pluribus' chips behind (not yet committed) before the decision
  eff_stack      min(stack, max chips behind among opponents still in the hand)
  spr            eff_stack / pot
  amount_to      for bets/raises: the total the street bet is made *to* (PHH 'cbr' amount)
  bet_frac_pot   for bets/raises: raise increment over the current max bet divided by the pot
                 after calling, i.e. (amount_to - max_bet) / (pot + to_call). 1.0 = pot-sized.
                 For a bet into an unbet pot this is just amount / pot.
  history        prior actions in the hand, street-separated by '|board|' markers; private hole-card
                 deals are excluded so the history only holds what Pluribus could observe.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pokerkit import HandHistory

BB = 100
POSITIONS = ["SB", "BB", "UTG", "MP", "CO", "BTN"]  # seat p1..p6 in these files
STREETS = ["preflop", "flop", "turn", "river"]
BOT = "Pluribus"


@dataclass
class Snap:
    """Immutable copy of the state fields we need (pokerkit mutates one State in place)."""

    street_index: int | None
    bets: list[int]
    stacks: list[int]
    total_pot_amount: int
    statuses: list[bool]
    payoffs: list[int]

    @classmethod
    def of(cls, st) -> "Snap":
        return cls(
            st.street_index,
            [int(x) for x in st.bets],
            [int(x) for x in st.stacks],
            int(st.total_pot_amount),
            list(st.statuses),
            [int(x) for x in st.payoffs],
        )


@dataclass
class HandResult:
    decisions: list[dict]
    hand: dict


def parse_hand(hh: HandHistory, hand_id: str) -> HandResult:
    players = list(hh.players)
    if players.count(BOT) != 1:
        raise ValueError(f"{hand_id}: expected exactly one {BOT}, got {players}")
    if list(hh.blinds_or_straddles) != [50, 100, 0, 0, 0, 0]:
        raise ValueError(f"{hand_id}: unexpected blinds {hh.blinds_or_straddles}")
    seat = players.index(BOT)
    n = len(players)

    hole: dict[int, str] = {}
    board = ""
    history: list[str] = []
    street_raises = 0          # voluntary bets/raises on the current street (blinds excluded)
    last_street = 0
    aggressor_by_street: dict[int, int] = {}  # street -> seat of last bettor/raiser
    decisions: list[dict] = []
    prev = None

    for live_state, action in hh.state_actions:
        state = Snap.of(live_state)
        if action is None:
            prev = state
            continue
        parts = action.split()
        if parts[0] == "d":
            if parts[1] == "dh":
                hole[int(parts[2][1:]) - 1] = parts[3]
            elif parts[1] == "db":
                board += parts[2]
                history.append(f"|{parts[2]}|")
            prev = state
            continue

        actor = int(parts[0][1:]) - 1
        verb = parts[1]
        if verb == "sm":  # show / muck at showdown: not a decision
            prev = state
            continue

        street = prev.street_index
        if street != last_street:
            street_raises = 0
            last_street = street

        if actor == seat:
            bets = list(prev.bets)
            stacks = list(prev.stacks)
            max_bet = max(bets)
            to_call = min(max_bet - bets[seat], stacks[seat])
            pot = int(prev.total_pot_amount)
            active_opps = [i for i in range(n) if i != seat and prev.statuses[i]]
            eff = min(stacks[seat], max(stacks[i] for i in active_opps)) if active_opps else 0

            amount_to = None
            bet_frac = None
            if verb == "f":
                act = "fold"
            elif verb == "cc":
                act = "check" if to_call == 0 else "call"
            elif verb == "cbr":
                amount_to = int(parts[2])
                act = "bet" if max_bet == 0 else "raise"
                bet_frac = (amount_to - max_bet) / (pot + to_call)
            else:
                raise ValueError(f"{hand_id}: unknown verb {verb!r}")

            all_in = (
                (verb == "cbr" and amount_to == stacks[seat] + bets[seat])
                or (verb == "cc" and to_call > 0 and to_call >= stacks[seat])
            )
            prev_aggr = aggressor_by_street.get(street - 1) if street > 0 else None
            decisions.append(
                {
                    "hand_id": hand_id,
                    "seat": seat + 1,
                    "position": POSITIONS[seat],
                    "street": STREETS[street],
                    "street_index": street,
                    "pot": pot,
                    "to_call": int(to_call),
                    "stack": int(stacks[seat]),
                    "eff_stack": int(eff),
                    "spr": eff / pot if pot else None,
                    "n_active": len(active_opps) + 1,
                    "raises_before": street_raises,
                    "history": ",".join(history),
                    "board": board,
                    "hole_cards": hole.get(seat),
                    "action": act,
                    "amount_to": amount_to,
                    "bet_frac_pot": bet_frac,
                    "all_in": bool(all_in),
                    "preflop_aggressor": aggressor_by_street.get(0) == seat,
                    "prev_street_aggressor_seat": (prev_aggr + 1) if prev_aggr is not None else None,
                    # prev-street aggressor still in the hand and acts after Pluribus on this street
                    # (post-flop action order is seat order p1..p6)
                    "prev_aggressor_acts_after": bool(
                        prev_aggr is not None
                        and prev_aggr != seat
                        and prev.statuses[prev_aggr]
                        and prev_aggr > seat
                    ),
                }
            )

        if verb == "cbr":
            street_raises += 1
            aggressor_by_street[street] = actor
        history.append(action)
        prev = state

    finishing = list(hh.finishing_stacks)
    starting = list(hh.starting_stacks)
    hole_ok = len(hole) == n and all(len(c) == 4 and "?" not in c for c in hole.values())
    payoff = finishing[seat] - starting[seat]
    final_payoffs = list(prev.payoffs) if prev is not None else None
    hand = {
        "hand_id": hand_id,
        "seat": seat + 1,
        "position": POSITIONS[seat],
        "hole_cards": hole.get(seat),
        "all_hole_cards_present": hole_ok,
        "n_hole_cards_logged": len(hole),
        "net_chips": float(payoff),  # logs contain half-chip split pots
        "pokerkit_payoff": int(final_payoffs[seat]) if final_payoffs else None,
        "n_decisions": len(decisions),
        "opponents": ",".join(p for p in players if p != BOT),
    }
    return HandResult(decisions, hand)


def hand_files(root: Path) -> list[Path]:
    def key(p: Path):
        return (p.parent.name, int(p.stem))

    return sorted(root.glob("*/*.phh"), key=key)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--data-dir", type=Path, default=here.parent / "data")
    args = ap.parse_args(argv)
    root = args.data_dir / "phh-dataset" / "data" / "pluribus"
    files = hand_files(root)
    print(f"found {len(files)} hand files under {root}")
    if not 9_500 <= len(files) <= 10_500:
        print("hand count is not ~10,000: stopping", file=sys.stderr)
        return 2

    decisions, hands = [], []
    for i, f in enumerate(files):
        with f.open("rb") as fh:
            hh = HandHistory.load(fh)
        r = parse_hand(hh, f"{f.parent.name}/{f.stem}")
        decisions.extend(r.decisions)
        hands.append(r.hand)
        if (i + 1) % 2000 == 0:
            print(f"  parsed {i + 1}")

    d = pd.DataFrame(decisions)
    h = pd.DataFrame(hands)
    for col in ["amount_to", "prev_street_aggressor_seat"]:
        d[col] = d[col].astype("Int64")
    d.to_parquet(args.data_dir / "pluribus_decisions.parquet", index=False)
    h.to_parquet(args.data_dir / "pluribus_hands.parquet", index=False)

    # pokerkit replays with integer chips; the logs split odd chips in half (e.g. 137.5)
    mismatch = ((h["net_chips"] - h["pokerkit_payoff"]).abs() > 0.5).sum()
    halves = (h["net_chips"] % 1 != 0).sum()
    print(f"decisions: {len(d)}, hands: {len(h)}")
    print(f"hands with all 6 hole cards logged: {h['all_hole_cards_present'].sum()} / {len(h)}")
    print(f"finishing_stacks vs pokerkit-replayed payoff mismatches (>0.5 chip): {mismatch}")
    print(f"hands with half-chip split-pot results: {halves}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
