# pluribus-lite

## Goal
An educational, scaled-down reimplementation of Pluribus (Brown & Sandholm, *Science* 2019,
"Superhuman AI for multiplayer poker"). The aim is to understand and reproduce the method at
small scale: CFR/MCCFR, abstraction, blueprint training and depth-limited search. Matching the
original's strength is not a goal.

## Hard constraints
- **Offline research only.** No integration with any online poker site, no screen reading,
  no automation of real games. Nothing in this repo may connect to, observe or act in a live game.
- **Do not touch** the existing Playground repo, or the Windows scheduled task that runs
  `routine.py` at 22:45 on weekdays.
- **Ask before installing system-wide software** (Rust toolchains, WSL2, Python versions,
  Visual Studio Build Tools, etc.).
- **Engine is pure Rust.** No crates that need a C/C++ build step, so the user-local
  `stable-x86_64-pc-windows-gnu` toolchain stays sufficient and the same code builds unchanged
  on Linux/WSL2.

## Milestones
- **M0: data + stats.** Parse the Pluribus hands from `uoftcprg/phh-dataset` into
  `data/pluribus_decisions.parquet`. Descriptive stats in `docs/m0_stats.md`.
- **M1: correctness harness.** CFR and MCCFR (external sampling, linear discounting,
  negative-regret pruning) validated on Kuhn and Leduc, with exploitability computed via
  best response.
- **M2: 6-max abstraction.** Card buckets and bet sizes.
- **M3: blueprint training.**
- **M4: depth-limited search, and evaluation against the Pluribus hands.**

## Conventions
- Every module has tests (`cargo test` in `engine/`, `pytest` in `analysis/`).
- Fixed RNG seeds everywhere. Seeds are explicit parameters and get logged with results.
- Every long run saves checkpoints periodically and can resume from the latest one.
- `data/` is gitignored. Generated datasets are reproducible from scripts in `analysis/`.
- **Linux-portable.** Code must build and run unchanged on Linux (HPC, SLURM): std-only Rust,
  `PathBuf` paths, no Windows APIs, no shell-specific behaviour in binaries. Long runs take a
  `--stop-file` and checkpoint on request. See `scripts/slurm_train.sh` (96 h jobs chained by
  checkpoint).
- Long local jobs run at below-normal priority and never longer than 2 hours unattended.

## Commands
- Engine tests: `cd engine && cargo test` (cargo is at `~/.cargo/bin`, not on PATH)
- Python tests: `cd analysis && .venv/Scripts/python -m pytest -q tests`
- M2 estimator: `cargo run --release --bin m2_estimate -- nlhe/configs/m2_11g.cfg`
- Evaluator benchmark: `cargo run --release --bin eval_bench`

## Layout
- `analysis/`: Python. Venv at `analysis/.venv` (Python 3.13). Packages: pokerkit, pandas,
  pyarrow, matplotlib, pytest. Run `analysis/.venv/Scripts/python`.
- `engine/`: Rust cargo workspace.
- `docs/`: notes, stats reports, and the paper supplement PDF.
- `data/`: raw dataset clone and derived parquet files (gitignored).

## Machine
- CPU: Intel Core i7-10750H @ 2.60 GHz, 6 cores / 12 threads (laptop).
- RAM: **16 GB total (15.8 GB usable)**, 2 x 8 GB DDR4 dual-channel, running at 2933 MT/s.
  This is the binding constraint: blueprint tables plus the abstraction must fit well under
  ~10 GB. Size the M2 abstraction for that limit, which means far coarser than the paper's.
- Disk: C: only, ~194 GB free of 952 GB at setup.
- OS: Windows 11 Pro 10.0.26300. WSL2 not installed.
- Python: 3.13.1 (default via `py`), plus 3.12 and 3.10. The `python3` command is the
  Microsoft Store stub, so don't use it.
- Rust: rustup with `stable-x86_64-pc-windows-gnu` (rustc 1.99.1), user-local in
  `~/.cargo` and `~/.rustup`. **PATH was not modified**, so call
  `~/.cargo/bin/cargo` directly or prepend it to PATH in your shell session.
- No Visual Studio or MSVC build tools are installed.
