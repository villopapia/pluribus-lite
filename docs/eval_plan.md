# Evaluation plan

Plan only: nothing here is implemented yet beyond what is marked **(exists)**. It applies to
every blueprint we train (pilot and faithful) and later to M4 search.

The plan has to work around three facts:
1. We cannot play elite professionals as the paper did. Our ground truth is the 10,000
   logged hands of Pluribus (`data/pluribus_decisions.parquet`, all hole cards known).
2. Six-player poker has no exploitability-based convergence guarantee (supplement p. 15;
   thesis §6.6), so "closer to equilibrium" cannot be measured directly.
3. Win rates are extremely noisy (our raw estimate of Pluribus' own win rate has a standard
   error of 8.8 bb/100, `docs/m0_stats.md`), so every win-rate number needs variance
   reduction.

## 1. Behavioural similarity to the Pluribus logs

**Goal:** quantify how much our blueprint (and later blueprint + search) plays like Pluribus,
on the decisions Pluribus actually faced.

### 1.1 Aggregate statistics

For each of our agents, compute the same statistics as `docs/m0_stats.md` and
`docs/m0_donk.md` **(exists, for the logs)** from self-play hands of our agent:
- VPIP and PFR by position; 3-bet % by position
- flop c-bet % (heads-up / multiway)
- donk-bet % by street (strict definition)
- action mix by street
- bet-size distributions by street (same bins as `m0_stats.md`)

Report each next to Pluribus' value with a 95% interval. Logged-side intervals are binomial
or bootstrap over hands; agent-side intervals are bootstrap over self-play hands, with enough
hands that they are much narrower than the logged side.

**Caveat:** Pluribus' post-flop play comes from real-time search (p. 19), so post-flop
statistics compare our *blueprint* to Pluribus' *search* until M4 exists. Preflop statistics
are the fair comparison for the blueprint.

### 1.2 Average log-likelihood of Pluribus' actual actions

For every logged Pluribus decision (15,169 rows):
1. Map the situation into our abstraction:
   - Cards: preflop class / flop and turn table bucket / river OCHS bucket, from Pluribus'
     known hole cards and the board.
   - Betting history: map onto our action tree, translating off-tree sizes with the paper's
     pseudo-harmonic mapping [39]. Record how often translation was needed and by how much.
2. Read our policy's distribution over abstract actions, and map Pluribus' actual action to
   an abstract action the same way.
3. Score log p(action), with a floor ε = 10⁻⁴ so a single zero cannot dominate. Report the
   mean per street and per position, plus:
   - **top-1 agreement** (Pluribus' action is our most likely action);
   - **fold/call/raise agreement** (ignoring sizes);
   - for raises, the **size-bucket agreement**.

**Baselines:** the uniform policy over the same abstract actions, and a "frequency" policy that
plays each node's empirical action frequencies from the logs (an upper reference that
overfits).

Run it on preflop decisions first (blueprint only); post-flop after M4.

## 2. Self-play head-to-head

**(exists: `h2h` binary, duplicate format: one copy of A in each seat against five copies of
B, common random numbers, 6 seat rotations per deal.)**

| comparison | purpose |
|---|---|
| checkpoint t vs checkpoint t/2, t/4 … within one run | learning curve; should be positive and shrinking |
| final pilot vs final faithful (in the faithful game, pilot mapped through action translation) | cost of the pilot's reduced abstraction |
| same config, different seeds (if compute allows) | run-to-run variation (section 4) |
| blueprint vs blueprint + search (M4) | value of search, the paper's main claim |

- **Units:** mbb/hand (= mbb/game in the paper).
- **Uncertainty:** report the standard error and a one-tailed t-test at 95%, as in the paper
  (p. 889 of the main paper).
- **Size:** at least 200,000 deals × 6 seats per comparison; increase until the standard error
  is below 1/5 of the effect being claimed.

## 3. Variance reduction for any win-rate figure

The paper used AIVAT [44] (Burch et al., AAAI 2018, `docs/papers/Burch2018-AAAI-AIVAT.pdf`),
adapted to more than two players (supplement pp. 7–9). Plan:
1. Implement the AIVAT estimator for our agents using our own blueprint as the value function:
   - **Chance correction:** subtract the expected value change of each chance event (deal, flop,
     turn, river) under the blueprint's own value estimates.
   - **Action correction:** for our agent's own actions, subtract the expected value change of
     its sampled action under its known policy.
   - Both corrections have zero mean, so the estimator is unbiased.
2. **Validate on self-play:** the AIVAT estimate must agree with the raw estimate within
   error, with a much smaller variance. Report the variance-reduction factor.
3. **Apply it to the logged hands, with care.** We know all hole cards, but we do not have
   Pluribus' actual policy (only our approximation). So a log-based AIVAT is "Pluribus'
   result corrected with our value estimates" and must be labelled as such. It is not the
   paper's 48 mbb/game.

## 4. Run-to-run variation

If compute allows a second seed for any configuration:
- head-to-head between seeds (should be ≈ 0, within error);
- the spread of the section 1 statistics between seeds, which separates real differences
  from seed noise;
- strategy distance between seeds on sampled infosets (same L1 metric as the training
  metric `strat_change`).

With one seed, all comparisons in sections 1–2 must be presented as single-run results.

## 5. Convergence proxies during training **(exists in `metrics.tsv`)**

These are diagnostics, not success criteria:
- strategy change between snapshots
- average positive regret per infoset (also divided by iterations)
- share of infosets and action sequences reached; the paper's equivalent is 413.5M of 664.8M
  sequences encountered (p. 13), so this is directly comparable for `faithful.cfg`
- nodes per iteration, river-reach rate, pruned actions per iteration

## Order of work

1. Section 1.2 (preflop log-likelihood) and section 1.1 preflop statistics, on the pilot
   blueprint.
2. AIVAT (section 3), validated on self-play.
3. Section 2 comparisons with AIVAT-corrected numbers.
4. Post-flop parts of section 1 after M4.
