"""D-ACT-5: which "fraction of the pot" convention do Pluribus' logged raise sizes follow?

The supplement (p. 12) says raise sizes are "fractions of the pot" without defining the pot.
Two readings:

* **A, pot after calling** (our default): raise increment = f x (pot + amount to call).
* **B, pot before calling**: raise increment = f x pot.

For bets (nothing to call) both are identical, so only **raises** (to_call > 0, not all-in)
can discriminate. If sizes were chosen as round fractions under one convention, Pluribus'
raises should sit tightly on round fractions under that convention and be smeared under the
other. We measure, per convention, how tightly raises cluster on the calibration grid
(`calibrate_sizes.GRID`) and, grid-free, how concentrated their distribution is.

Usage:  .venv/Scripts/python pot_convention.py [--decisions ../data/pluribus_decisions.parquet]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from calibrate_sizes import GRID


def raise_fractions(d: pd.DataFrame) -> pd.DataFrame:
    """Pluribus' non all-in raises with the fraction under both conventions."""
    r = d[(d.action == "raise") & ~d.all_in & (d.to_call > 0)].copy()
    inc = r.bet_frac_pot * (r.pot + r.to_call)  # chips added over the current max bet
    r["frac_after_call"] = inc / (r.pot + r.to_call)
    r["frac_before_call"] = inc / r.pot
    return r


def tightness(x: np.ndarray, tol: float = 0.02) -> dict[str, float]:
    """Clustering of fractions x: on the grid (share within `tol` relative distance of a grid
    size, median relative distance) and grid-free (entropy of 0.01-wide bins in log space,
    number of such bins holding 80% of the mass)."""
    x = x[np.isfinite(x) & (x > 0)]
    rel = np.abs(GRID[None, :] / x[:, None] - 1.0).min(1)
    bins = np.round(np.log(x) / 0.01).astype(int)
    p = pd.Series(bins).value_counts(normalize=True).to_numpy()
    cum = np.cumsum(np.sort(p)[::-1])
    return {
        "n": len(x),
        "on_grid_2pct": float((rel <= tol).mean()),
        "median_rel_dist": float(np.median(rel)),
        "entropy_bits": float(-(p * np.log2(p)).sum()),
        "bins_for_80pct": int(np.searchsorted(cum, 0.8) + 1),
    }


def report(r: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("all raises", r)]
    groups += [(f"{s} depth {k}", g) for (s, k), g in r.groupby(["street", "raises_before"]) if len(g) >= 30]
    for name, g in groups:
        for conv in ["frac_after_call", "frac_before_call"]:
            rows.append({"group": name, "convention": conv, **tightness(g[conv].to_numpy())})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", default="../data/pluribus_decisions.parquet")
    a = ap.parse_args()
    d = pd.read_parquet(a.decisions)
    r = raise_fractions(d)
    pd.set_option("display.width", 200)
    print(report(r).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
