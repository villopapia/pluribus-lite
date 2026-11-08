"""M0: descriptive stats of Pluribus' play -> docs/m0_stats.md (+ docs/m0_bet_sizes.png).

Usage:  .venv/Scripts/python -I m0_stats.py

Stat definitions (per hand, from Pluribus' own decisions):
  VPIP   hands with a voluntary preflop call or raise / hands with >= 1 preflop decision
         (walks in the BB, where Pluribus never acts, are excluded from the denominator)
  PFR    hands with a preflop raise / same denominator
  3-bet  first preflop decision facing exactly one voluntary raise: share that re-raise
  C-bet  Pluribus was the last preflop raiser and makes the first bet-or-check decision on
         an unbet flop: share that bet
  Donk   Pluribus was NOT the preflop aggressor, the preflop aggressor is still in and acts
         after Pluribus on the flop, the flop is unbet: share that bet
  Win rate  mean net chips per hand / 100 chips * 100 hands = bb/100 (raw, no AIVAT)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BB = 100
POSITIONS = ["UTG", "MP", "CO", "BTN", "SB", "BB"]  # preflop action order
STREETS = ["preflop", "flop", "turn", "river"]
# Pluribus' post-flop sizes are discrete (1/4, 1/2, 3/4, 1, 1.5, 2, 3 pot, rounded to whole chips),
# so bins are centred on those sizes rather than having edges at them.
POST_BINS = [0, 0.2, 0.3, 0.4, 0.6, 0.85, 1.25, 1.75, 2.5, np.inf]
POST_LABELS = ["<20%", "~25%", "~33%", "~50%", "~75%", "~100%", "~150%", "~200%", ">250%"]
PRE_BINS = [0, 2.25, 2.75, 3.25, 4.5, 7, 12, 25, np.inf]
PRE_LABELS = ["<=2.25bb", "2.25-2.75", "2.75-3.25", "3.25-4.5", "4.5-7", "7-12", "12-25", ">25bb"]

# dataviz palette (single series, single hue)
INK, INK2, SURFACE, BLUE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6", "#e4e3df"


def first_per_hand(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("hand_id", sort=False).head(1)


def preflop_table(d: pd.DataFrame, hands: pd.DataFrame) -> pd.DataFrame:
    pre = d[d.street == "preflop"]
    g = pre.groupby("hand_id")
    per_hand = pd.DataFrame(
        {
            "position": g.position.first(),
            "vpip": g.action.apply(lambda a: a.isin(["call", "raise"]).any()),
            "pfr": g.action.apply(lambda a: (a == "raise").any()),
        }
    )
    tb_opp = first_per_hand(pre[pre.raises_before == 1])
    tb = (
        tb_opp.assign(is3=(tb_opp.action == "raise").astype("int64"))
        .groupby("position")
        .is3.agg(threebet_opps="size", threebets="sum")
        .astype("float64")
    )
    out = per_hand.groupby("position").agg(hands_acted=("vpip", "size"), vpip=("vpip", "mean"), pfr=("pfr", "mean"))
    out["hands_dealt"] = hands.groupby("position").size()
    out = out.join(tb)
    out["threebet"] = out.threebets / out.threebet_opps
    n3 = float(len(tb_opp))
    k3 = float((tb_opp.action == "raise").sum())
    tot = pd.Series(
        {
            "hands_acted": len(per_hand),
            "vpip": per_hand.vpip.mean(),
            "pfr": per_hand.pfr.mean(),
            "hands_dealt": len(hands),
            "threebet_opps": n3,
            "threebets": k3,
            "threebet": k3 / n3 if n3 else float("nan"),
        },
        name="All",
    )
    out = out.reindex(POSITIONS)
    return pd.concat([out, tot.to_frame().T])


def flop_spots(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    flop_first = first_per_hand(d[d.street == "flop"])
    unbet = flop_first[flop_first.raises_before == 0]
    cbet = unbet[unbet.preflop_aggressor]
    donk = unbet[(~unbet.preflop_aggressor) & unbet.prev_aggressor_acts_after]
    return cbet, donk


def rate(df: pd.DataFrame) -> tuple[int, int, float]:
    n = len(df)
    k = int((df.action == "bet").sum())
    return k, n, (k / n if n else float("nan"))


def bet_size_tables(d: pd.DataFrame) -> dict[str, pd.Series]:
    agg = d[d.action.isin(["bet", "raise"])]
    out = {}
    pre = agg[agg.street == "preflop"]
    out["preflop"] = pd.cut(pre.amount_to / BB, PRE_BINS, labels=PRE_LABELS).value_counts().reindex(PRE_LABELS)
    for s in STREETS[1:]:
        x = agg[agg.street == s].bet_frac_pot
        out[s] = pd.cut(x, POST_BINS, labels=POST_LABELS).value_counts().reindex(POST_LABELS)
    return out


def plot_bet_sizes(tables: dict[str, pd.Series], path: Path) -> None:
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), facecolor=SURFACE)
    for ax, s in zip(axes.flat, STREETS):
        t = tables[s]
        share = t / t.sum() * 100
        ax.set_facecolor(SURFACE)
        bars = ax.bar(range(len(t)), share.values, color=BLUE, width=0.8, edgecolor=SURFACE, linewidth=2)
        top = share.max()
        for b, v, c in zip(bars, share.values, t.values):
            if v >= 0.12 * top:  # selective direct labels: only bars big enough to matter
                ax.text(b.get_x() + b.get_width() / 2, v + top * 0.02, f"{v:.0f}%", ha="center",
                        va="bottom", fontsize=8, color=INK2)
        ax.set_xticks(range(len(t)), t.index, rotation=35, ha="right")
        ax.set_ylim(0, top * 1.15)
        unit = "raise-to size (bb)" if s == "preflop" else "bet/raise increment as % of pot after calling"
        ax.set_title(f"{s.capitalize()}  (n = {int(t.sum())})", loc="left", fontsize=10, color=INK)
        ax.set_xlabel(unit, fontsize=8)
        ax.set_ylabel("% of bets/raises", fontsize=8)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(length=0)
    fig.suptitle("Pluribus bet and raise sizes by street", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{100 * x:.1f}%"


def build_report(d: pd.DataFrame, h: pd.DataFrame, png_name: str) -> str:
    L: list[str] = []
    n = len(h)
    L += [
        "# M0: Pluribus descriptive stats",
        "",
        "Generated by `analysis/m0_stats.py` from `data/pluribus_decisions.parquet` "
        "(built by `analysis/m0_parse.py` from `uoftcprg/phh-dataset`, `data/pluribus/`). "
        "**Dataset validated** against Science Data File S1: 10,000 / 10,000 hands identical in all "
        "fields (`docs/s1_crosscheck.md`).",
        "",
        f"- Hands: **{n:,}**, all with stacks reset to 100 bb (10,000 chips), blinds 50/100.",
        f"- Pluribus decisions: **{len(d):,}** "
        f"({', '.join(f'{s} {int((d.street == s).sum()):,}' for s in STREETS)}).",
        f"- Hole cards: all 6 players' hole cards are logged in "
        f"**{int(h.all_hole_cards_present.sum()):,} / {n:,}** hands, including hands that "
        "ended before showdown.",
        "",
    ]

    # win rate
    net = h.net_chips
    wr = net.mean() / BB * 100
    se = net.std(ddof=1) / np.sqrt(n) / BB * 100
    L += [
        "## Raw win rate (no variance reduction)",
        "",
        f"**{wr:+.2f} bb/100** (standard error {se:.2f} bb/100, 95% CI "
        f"{wr - 1.96 * se:+.1f} to {wr + 1.96 * se:+.1f}). Total {net.sum() / BB:+,.1f} bb over {n:,} hands.",
        "",
        "The paper reports 48 mbb/game (= 4.8 bb/100) with AIVAT. The raw figure is much noisier: "
        "its standard error is several bb/100, so the two are not directly comparable.",
        "",
        "| position | hands | net bb | bb/100 | s.e. |",
        "|---|---:|---:|---:|---:|",
    ]
    for pos in POSITIONS:
        x = h[h.position == pos].net_chips
        L.append(
            f"| {pos} | {len(x):,} | {x.sum() / BB:+,.1f} | {x.mean():+.1f} | "
            f"{x.std(ddof=1) / np.sqrt(len(x)):.1f} |"
        )
    L.append("")

    # preflop
    t = preflop_table(d, h)
    L += [
        "## Preflop: VPIP, PFR, 3-bet by position",
        "",
        "VPIP/PFR denominators exclude walks (BB hands folded to Pluribus, where it never acts).",
        "",
        "| position | hands dealt | hands acted | VPIP | PFR | 3-bet opps | 3-bet |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for pos, r in t.iterrows():
        L.append(
            f"| {pos} | {int(r.hands_dealt):,} | {int(r.hands_acted):,} | {pct(r.vpip)} | {pct(r.pfr)} | "
            f"{int(r.threebet_opps) if not pd.isna(r.threebet_opps) else 0:,} | {pct(r.threebet)} |"
        )
    L.append("")

    cbet, donk = flop_spots(d)
    ck, cn, cr = rate(cbet)
    dk, dn, dr = rate(donk)
    L += [
        "## Flop: continuation bets and donk bets",
        "",
        "| stat | opportunities | bets | rate |",
        "|---|---:|---:|---:|",
        f"| C-bet (Pluribus was preflop aggressor, flop unbet when it acts) | {cn:,} | {ck:,} | {pct(cr)} |",
        f"| Donk bet (bets into the preflop aggressor, who acts later) | {dn:,} | {dk:,} | {pct(dr)} |",
        "",
        "The donk row uses a loose v1 definition (flop only; Pluribus acts before the preflop "
        "aggressor, not necessarily first). See `m0_donk.md` for the strict definition (called the "
        "last bet, first to act, bets) by street, compared with the human players.",
        "",
    ]
    # c-bet heads-up vs multiway
    hu = cbet[cbet.n_active == 2]
    mw = cbet[cbet.n_active > 2]
    L += [
        f"C-bet heads-up: {pct(rate(hu)[2])} (n = {len(hu)}); multiway: {pct(rate(mw)[2])} (n = {len(mw)}).",
        "",
    ]

    # action mix
    L += ["## Action mix by street", "", "| street | n | fold | check | call | bet | raise |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for s in STREETS:
        x = d[d.street == s].action
        L.append(
            f"| {s} | {len(x):,} | "
            + " | ".join(pct((x == a).mean()) for a in ["fold", "check", "call", "bet", "raise"])
            + " |"
        )
    L.append("")

    # bet sizes
    tables = bet_size_tables(d)
    L += [
        "## Bet and raise sizes by street",
        "",
        f"![Pluribus bet and raise sizes by street]({png_name})",
        "",
        "Preflop is binned by the raise-to amount in big blinds. Post-flop is binned by "
        "`bet_frac_pot` = (raise-to amount - current bet) / (pot after calling), so 100% is a "
        "pot-sized bet or raise. Counts:",
        "",
    ]
    L += ["| preflop raise-to | count |", "|---|---:|"]
    L += [f"| {k} | {int(v)} |" for k, v in tables["preflop"].items()]
    L += ["", "| pot fraction | flop | turn | river |", "|---|---:|---:|---:|"]
    for lab in POST_LABELS:
        L.append(f"| {lab} | " + " | ".join(str(int(tables[s][lab])) for s in STREETS[1:]) + " |")
    agg = d[d.action.isin(["bet", "raise"])]
    L += ["", "| street | median | mean | all-in share |", "|---|---:|---:|---:|"]
    for s in STREETS:
        x = agg[agg.street == s]
        if s == "preflop":
            L.append(f"| preflop (bb) | {x.amount_to.median() / BB:.2f} | {x.amount_to.mean() / BB:.2f} | {pct(x.all_in.mean())} |")
        else:
            L.append(f"| {s} (pot) | {x.bet_frac_pot.median():.2f} | {x.bet_frac_pot.mean():.2f} | {pct(x.all_in.mean())} |")
    L.append("")
    return "\n".join(L)


def main() -> int:
    here = Path(__file__).resolve().parent
    data = here.parent / "data"
    docs = here.parent / "docs"
    d = pd.read_parquet(data / "pluribus_decisions.parquet")
    h = pd.read_parquet(data / "pluribus_hands.parquet")
    png = "m0_bet_sizes.png"
    plot_bet_sizes(bet_size_tables(d), docs / png)
    (docs / "m0_stats.md").write_text(build_report(d, h, png), encoding="utf-8")
    print(f"wrote {docs / 'm0_stats.md'} and {docs / png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
