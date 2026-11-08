import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subgame_roots import roots_of  # noqa: E402


def test_roots_track_contributions_folds_and_streets():
    acts = [
        "d dh p1 3c9s", "p3 f", "p4 f", "p5 cbr 225", "p6 f", "p1 cc", "p2 cc",  # 3-way flop
        "d db 2c3d4h", "p1 cc", "p2 cbr 300", "p5 cc", "p1 f",                   # bet, call, fold
        "d db 5s", "p2 cc", "p5 cc",                                             # heads-up turn
        "d db 6s",
    ]
    r = roots_of(acts, bot_seat=4)
    assert r == [
        (1, 0b101100, [225, 225, 0, 0, 225, 0]),
        (2, 0b101101, [225, 525, 0, 0, 525, 0]),
        (3, 0b101101, [225, 525, 0, 0, 525, 0]),
    ]
    # Pluribus (seat 0) folded on the flop: no later roots
    assert [x[0] for x in roots_of(acts, bot_seat=0)] == [1]


def test_no_root_when_fewer_than_two_players_can_act():
    acts = ["p3 cbr 10000", "p4 f", "p5 f", "p6 f", "p1 f", "p2 cc", "d db 2c3d4h", "d db 5s", "d db 6s"]
    assert roots_of(acts, bot_seat=1) == []
