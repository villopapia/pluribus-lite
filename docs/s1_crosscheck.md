# Data File S1 cross-check (D-DATA-1)

Done 2026-11-08 with `analysis/s1_crosscheck.py` (tests: `analysis/tests/test_s1_crosscheck.py`).

## Source
- `data/aay2400_data_file_s1.zip`, SHA-256 `f1afeea99645f96db40d0fafc19e3da37fbab3ab59456db9f2e6039abd880f0c`
  (Science "Data File S1", supplied by the user), unzipped to `data/s1/5H1AI_logs/`.
- 92 logs `sample_game_<game>.log` (games 30–118 plus "b" sessions) + `README.txt`; **10,000 hands**,
  the 5 humans + 1 AI experiment only (the 1H+5AI hands are not in S1).

## Original log format (ACPC "STATE" lines, documented in the zip's README.txt)
`STATE:<index>:<actions>:<hole|hole|...>/<flop>/<turn>/<river>:<payoffs>:<players>`
- actions: rounds separated by `/`; `f` fold, `c` check/call, `r<N>` raise to **N = the raiser's
  total chips in the pot for the whole hand** (not per street);
- **all six hole-card pairs** are listed, folded players included;
- payoffs per player (8 hands have half-chip payoffs, e.g. 112.5, from split pots);
- players in seat order, first = small blind; header lines `#` give the ACPC game definition
  `holdem.nolimit.6p.game`, 10,000 hands and the seed per session.

## Result
| field | mismatching hands (of 10,000) |
|---|---:|
| players / seat order | 0 |
| hole cards (all six) | 0 |
| board | 0 |
| action sequence (amounts rebuilt as hand totals from PHH's per-street `cbr`) | 0 |
| payoffs (PHH finishing − starting stacks) | 0 |

**All 10,000 hands are identical**; every S1 hand has exactly one PHH file
(`data/pluribus/<game>/<index>.phh`), and file names map one to one. The half-chip payoffs are
preserved in PHH too. The comparator is tested to detect a mismatch in each field
(`test_every_field_mismatch_is_detected`). Conclusion: `uoftcprg/phh-dataset` is a lossless
conversion of S1, so M0 statistics and the calibration rest on the original logs.
