# M1: negative-regret pruning threshold

Reproduce: `cd engine && cargo run --release --bin prune_study` (seed 1, 2M iterations per row).

## How C is scaled

Paper values are from memory and **still need checking against the supplement PDF**.

| | Pluribus (paper) | Kuhn (ours) | Leduc (ours) |
|---|---:|---:|---:|
| chip unit | 1 chip = 0.01 bb | 1 ante | 1 ante |
| starting stack / max \|utility\| | 10,000 chips | 2 | 13 |
| C | -300,000,000 chips | -60,000 | -390,000 |
| C / max \|utility\| | -30,000 | -30,000 | -30,000 |
| C in max single-traversal regret increments (≈ 2 × max \|u\|) | ≈ 15,000 | 15,000 | 15,000 |
| prune threshold (iterations) | 200 min of an ~11,520-min run | 1.74% of run | 1.74% of run |

`MccfrConfig::pluribus_scaled` keeps **C / max |utility|** equal to the paper's ratio, and
keeps the schedule (prune start, LCFR end, discount interval) as the same *fractions of the
run*.

**Caveat.** What determines when pruning fires is C relative to how much regret one infoset
accumulates, which is roughly visits × typical increment. A Kuhn infoset is visited millions of
times in a run; a Pluribus infoset is visited far less often. So the same C/stack ratio makes
pruning fire *much earlier per infoset* in the toy games than it did in Pluribus. In the
scaled runs it does fire: the most negative final regret is below C (table below).

## Study: pruning frequency, all-pruned nodes, `always_explore_best`

"Pruned-traversal nodes" = traverser decision nodes visited during pruned traversals (95% of
traversals after the prune threshold). "All-pruned" = nodes where every action had regret <= C.

### Kuhn, 2M iterations

| C | flag | pruned-traversal nodes | all-pruned nodes | pruned actions | min regret at end | exploitability |
|---:|---|---:|---:|---:|---:|---:|
| -60,000 (scaled) | off | 4,562,988 | 0 | 1,367,984 | -73,450 | 0.000521 |
| -60,000 (scaled) | on | 4,562,988 | 0 | 1,367,984 | -73,450 | 0.000521 |
| -200 | off | 4,494,069 | 0 | 4,023,834 | -17,799 | 0.003219 |
| -200 | on | 4,494,069 | 0 | 4,023,834 | -17,799 | 0.003219 |
| -5 | off | 4,380,275 | 0 | 4,333,635 | -17,799 | 0.001159 |
| -5 | on | 4,380,275 | 0 | 4,333,635 | -17,799 | 0.001159 |

### Leduc, 2M iterations

| C | flag | pruned-traversal nodes | all-pruned nodes | pruned actions | min regret at end | exploitability |
|---:|---|---:|---:|---:|---:|---:|
| -390,000 (scaled) | off | 16,590,263 | 0 | 556,123 | -420,333 | 0.009355 |
| -390,000 (scaled) | on | 16,590,263 | 0 | 556,123 | -420,333 | 0.009355 |
| -1,300 | off | 10,194,938 | 0 | 9,529,232 | -50,191 | 0.106641 |
| -1,300 | on | 10,194,938 | 0 | 9,529,232 | -50,191 | 0.106641 |
| -32.5 | off | 7,959,628 | 0 | 9,688,676 | -53,945 | 0.054335 |
| -32.5 | on | 7,959,628 | 0 | 9,688,676 | -53,945 | 0.054335 |

## Findings

1. **All-pruned nodes never occurred** (0 of ~48M pruned-traversal node visits), so
   `always_explore_best` (default off) changed nothing, bit for bit.
2. **This is a theorem, not luck**, for any C < 0 with exact regret-matching updates:
   - With σ ∝ R⁺, the update satisfies Σ_{a: R_a>0} R_a·r_a = 0.
   - Hence Σ_{a: R_a>0} R_a·(R_a + r_a) = Σ R_a² > 0, so some positive regret stays positive.
   - Regrets start at 0, and the first update leaves either all zeros or some positive regret.
   - Linear discounting multiplies by d > 0, which preserves signs.
   - Pruned actions have σ = 0 whenever any regret is positive, so skipping them doesn't break the identity.

   So max_a R(I,a) ≥ 0 > C always. This could change with a **regret floor** that clamps
   values, **integer regrets** with rounding, or **C ≥ 0**. The flag stays as a safety net
   for those cases, and the counter stays so we would notice.
3. **Too-aggressive C hurts badly on Leduc** (0.107 vs 0.009 exploitability at C = -100 × max|u|).
   Pruned actions are re-evaluated only in the 5% unpruned traversals, so actions that become
   good later recover slowly. Kuhn is more forgiving.
4. In this implementation pruning gives little speed-up on these games (Leduc 378k vs 365k
   it/s unpruned in `bench`). Pluribus' gains come from huge subtrees under bad actions,
   which Leduc does not have.
