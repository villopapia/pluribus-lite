"""D-DATA-1: the S1 <-> PHH comparator must detect every kind of mismatch."""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s1_crosscheck import compare, parse_s1_line, phh_to_acpc  # noqa: E402

# hand 100/0 from S1 and its PHH conversion (both copied verbatim)
S1 = (
    "STATE:0:fr210ffcf/cc/cc/r440f:TcQc|8s4c|9c3d|Ah4h|Th5s|6c7s/7d5h9d/7c/Qh:310|-100|0|-210|0|0:"
    "MrBlue|MrBlonde|MrWhite|MrPink|MrBrown|Pluribus"
)
PHH = {
    "blinds_or_straddles": [50, 100, 0, 0, 0, 0],
    "starting_stacks": [10000] * 6,
    "actions": [
        "d dh p1 TcQc", "d dh p2 8s4c", "d dh p3 9c3d", "d dh p4 Ah4h", "d dh p5 Th5s", "d dh p6 6c7s",
        "p3 f", "p4 cbr 210", "p5 f", "p6 f", "p1 cc", "p2 f", "d db 7d5h9d", "p1 cc", "p4 cc",
        "d db 7c", "p1 cc", "p4 cc", "d db Qh", "p1 cbr 230", "p4 f",
    ],
    "players": ["MrBlue", "MrBlonde", "MrWhite", "MrPink", "MrBrown", "Pluribus"],
    "finishing_stacks": [10310, 9900, 10000, 9790, 10000, 10000],
}


def test_real_hand_matches_and_raise_amounts_are_cumulative():
    s, ours = parse_s1_line(S1), phh_to_acpc(PHH)
    assert ours["actions"] == "fr210ffcf/cc/cc/r440f"  # river 'cbr 230' = 210 + 230 for the hand
    assert compare(s, ours) == []


def test_every_field_mismatch_is_detected():
    s = parse_s1_line(S1)
    for field, mutate in [
        ("holes", lambda h: h["actions"].__setitem__(0, "d dh p1 TcQd")),
        ("board", lambda h: h["actions"].__setitem__(15, "d db 7s")),
        ("actions", lambda h: h["actions"].__setitem__(19, "p1 cbr 231")),
        ("payoffs", lambda h: h["finishing_stacks"].__setitem__(0, 10309)),
        ("players", lambda h: h["players"].__setitem__(5, "Bot")),
    ]:
        h = copy.deepcopy(PHH)
        mutate(h)
        assert field in compare(s, phh_to_acpc(h)), field
