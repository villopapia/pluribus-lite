"""D-ACT-5 analysis: conversion between the two pot conventions and the tightness metric."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pot_convention import raise_fractions, tightness  # noqa: E402


def test_conversion_between_conventions():
    # pot 300, to_call 100: a pot-sized raise after calling adds 400 chips -> 400/300 before calling
    d = pd.DataFrame(
        {
            "action": ["raise", "bet", "raise"],
            "all_in": [False, False, True],
            "to_call": [100, 0, 100],
            "pot": [300, 300, 300],
            "bet_frac_pot": [1.0, 0.5, 1.0],
        }
    )
    r = raise_fractions(d)
    assert len(r) == 1  # bets and all-ins excluded
    assert np.isclose(r.frac_after_call.iloc[0], 1.0)
    assert np.isclose(r.frac_before_call.iloc[0], 400 / 300)


def test_tightness_prefers_values_on_the_grid():
    on = tightness(np.array([0.5, 0.5, 1.0, 0.75, 1.0]))
    off = tightness(np.array([0.57, 0.81, 1.13, 0.66, 1.37]))
    assert on["on_grid_2pct"] == 1.0 and off["on_grid_2pct"] < 0.5
    assert on["entropy_bits"] < off["entropy_bits"]
