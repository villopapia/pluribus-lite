"""Two-arm summary of heads-up matches (M4 NLHE validation): each arm's mean mbb/hand with s.e.
and 95% CI (raw, AIVAT chance terms, full AIVAT), and the PAIRED per-deal difference (arm A −
arm B) over the deals both arms finished. Both arms must use the same seed, so a deal is the same
cards in both; the shared card luck then cancels in the difference.

    python hu_pair.py ../data/hu_long_final.tsv ../data/hu_long_avg.tsv

Input: `hu_match --out` TSVs (one row per deal; values in chips per hand, 1 chip = 10 mbb).
"""

import sys

import numpy as np
import pandas as pd

COLS = [("raw", "raw"), ("chance", "AIVAT chance terms"), ("full", "AIVAT full")]


def stats(x):
    """Mean, standard error and 95% CI half-width of a 1-D array."""
    x = np.asarray(x, dtype=float)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    return m, se, 1.96 * se


def load(path):
    df = pd.read_csv(path, sep="\t", dtype={"expl": str})
    return df.drop_duplicates("deal", keep="last").set_index("deal")


def report(a, b, name_a="A", name_b="B"):
    """Return the summary as text: each arm, then the paired difference on common deals."""
    out = []
    for name, df in [(name_a, a), (name_b, b)]:
        out.append(f"{name}: {len(df)} deals ({2 * len(df)} hands)")
        for c, label in COLS:
            m, se, h = stats(df[c] * 10)
            out.append(f"  {label:20} {m:+8.1f} mbb/hand, s.e. {se:6.1f}, 95% CI [{m - h:+.0f}, {m + h:+.0f}]")
    common = a.index.intersection(b.index)
    out.append(f"paired difference {name_a} - {name_b}: {len(common)} common deals")
    for c, label in COLS:
        d = (a.loc[common, c] - b.loc[common, c]) * 10
        m, se, h = stats(d)
        r = np.corrcoef(a.loc[common, c], b.loc[common, c])[0, 1] if len(common) > 2 else float("nan")
        out.append(f"  {label:20} {m:+8.1f} mbb/hand, s.e. {se:6.1f}, 95% CI [{m - h:+.0f}, {m + h:+.0f}] (per-deal correlation of the arms {r:.2f})")
    return "\n".join(out)


if __name__ == "__main__":
    pa, pb = sys.argv[1], sys.argv[2]
    print(report(load(pa), load(pb), pa, pb))
