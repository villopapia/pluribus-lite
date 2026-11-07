"""Tests for the M0 parser and stats on a hand-constructed PHH hand with known answers."""

import pandas as pd
import pytest
from pokerkit import HandHistory

import m0_parse
import m0_stats

# Pluribus is p3 (UTG) with AA: opens 2.5bb, BB calls; flop c-bet half pot, BB check-raises,
# Pluribus calls; turn BB bets 1000, Pluribus folds.
HAND = """variant = 'NT'
ante_trimming_status = true
antes = [0, 0, 0, 0, 0, 0]
blinds_or_straddles = [50, 100, 0, 0, 0, 0]
min_bet = 100
starting_stacks = [10000, 10000, 10000, 10000, 10000, 10000]
actions = ['d dh p1 2c3d', 'd dh p2 4h5s', 'd dh p3 AsAh', 'd dh p4 7c8d', 'd dh p5 9h9d', 'd dh p6 TcJd', 'p3 cbr 250', 'p4 f', 'p5 f', 'p6 f', 'p1 f', 'p2 cc', 'd db KdQh2s', 'p2 cc', 'p3 cbr 275', 'p2 cbr 900', 'p3 cc', 'd db 3s', 'p2 cbr 1000', 'p3 f']
hand = 1
players = ['A', 'B', 'Pluribus', 'D', 'E', 'F']
finishing_stacks = [9950, 11200, 8850, 10000, 10000, 10000]
"""


@pytest.fixture(scope="module")
def parsed():
    return m0_parse.parse_hand(HandHistory.loads(HAND), "t/1")


def test_decisions_basic(parsed):
    d = parsed.decisions
    assert [x["action"] for x in d] == ["raise", "bet", "call", "fold"]
    assert [x["street"] for x in d] == ["preflop", "flop", "flop", "turn"]
    assert all(x["position"] == "UTG" and x["hole_cards"] == "AsAh" for x in d)


def test_preflop_open(parsed):
    x = parsed.decisions[0]
    assert (x["pot"], x["to_call"], x["amount_to"], x["raises_before"]) == (150, 100, 250, 0)
    assert x["bet_frac_pot"] == pytest.approx(150 / 250)
    assert x["history"] == ""


def test_flop_cbet(parsed):
    x = parsed.decisions[1]
    assert (x["pot"], x["to_call"], x["amount_to"]) == (550, 0, 275)
    assert x["bet_frac_pot"] == pytest.approx(0.5)
    assert x["preflop_aggressor"] and x["n_active"] == 2
    assert x["eff_stack"] == 9750 and x["spr"] == pytest.approx(9750 / 550)
    assert x["board"] == "KdQh2s"
    assert x["history"].endswith("p2 cc,|KdQh2s|,p2 cc")
    assert "AsAh" not in x["history"]  # private deals excluded


def test_facing_checkraise_and_turn_fold(parsed):
    x = parsed.decisions[2]
    assert (x["pot"], x["to_call"], x["raises_before"]) == (1725, 625, 2)
    y = parsed.decisions[3]
    assert (y["pot"], y["to_call"], y["street"]) == (3350, 1000, "turn")
    assert y["prev_street_aggressor_seat"] == 2


def test_hand_level(parsed):
    h = parsed.hand
    assert h["net_chips"] == -1150
    assert h["pokerkit_payoff"] == -1150
    assert h["all_hole_cards_present"]


def test_stats_on_synthetic_hand(parsed):
    d = pd.DataFrame(parsed.decisions)
    h = pd.DataFrame([parsed.hand])
    t = m0_stats.preflop_table(d, h)
    assert t.loc["UTG", "vpip"] == 1 and t.loc["UTG", "pfr"] == 1
    cbet, donk = m0_stats.flop_spots(d)
    assert m0_stats.rate(cbet) == (1, 1, 1.0)
    assert len(donk) == 0
    sizes = m0_stats.bet_size_tables(d)
    assert sizes["flop"]["~50%"] == 1
    assert sizes["preflop"]["2.25-2.75"] == 1


def test_parse_rejects_missing_or_duplicate_bot():
    bad = HAND.replace("'Pluribus'", "'X'")
    with pytest.raises(ValueError):
        m0_parse.parse_hand(HandHistory.loads(bad), "bad")
