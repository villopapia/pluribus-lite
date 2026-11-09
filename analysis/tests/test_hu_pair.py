"""hu_pair: per-arm statistics and the paired difference on common deals."""

import numpy as np
import pandas as pd

import hu_pair


def frame(deals, raw):
    raw = np.asarray(raw, dtype=float)
    return pd.DataFrame({"deal": deals, "raw": raw, "chance": raw, "full": raw}).set_index("deal")


def test_stats_known_values():
    m, se, h = hu_pair.stats([1.0, 2.0, 3.0, 4.0])
    assert m == 2.5
    assert abs(se - np.std([1, 2, 3, 4], ddof=1) / 2) < 1e-12
    assert abs(h - 1.96 * se) < 1e-12


def test_shared_luck_cancels_in_the_paired_difference():
    rng = np.random.default_rng(0)
    luck = rng.normal(0, 1000, 500)  # same cards in both arms
    a = frame(range(500), luck + 5)
    b = frame(range(500), luck)
    text = hu_pair.report(a, b)
    # the arms are noisy, the difference is exactly +5 chips = +50 mbb with zero s.e.
    assert "paired difference A - B: 500 common deals" in text
    assert "raw                     +50.0 mbb/hand, s.e.    0.0" in text


def test_only_common_deals_are_paired():
    a = frame([0, 1, 2, 3], [1, 2, 3, 4])
    b = frame([2, 3, 4], [0, 0, 0])
    assert "2 common deals" in hu_pair.report(a, b)
