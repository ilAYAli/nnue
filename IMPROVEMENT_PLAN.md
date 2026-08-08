# NNUE Architecture Improvement Plan

## Current status (2026-08-02)

Champion: `enyo-1.32.0-rc10`, `continue_from enyo-1.31.0-rc57`.
Absolute vs the fixed external target `nn-0ee0657fb25e.nnue`: **elo=-139.8**
(ci=19.2, 4000 games, engine enyo_11ca4d7 - see benchmarks/stockfish-net.jsonl).
This is the number to beat in absolute terms; promotion itself is decided
parent-relative against rc10.

The prior `enyo-scratch-broad`/`enyo-scratch-long` lineage this document used
to describe is dead history: an isolated side branch (2026-07-05/06), never
benchmarked against Stockfish, never adopted as continue_from by the main
numbered lineage, and superseded entirely by the `enyo-1.0.0` -> `enyo-1.33.0`
chain. See STATUS.md for the verified lineage table and its one known gap
(`enyo-1.30.0-rc3`'s own promotion commit is unrecoverable).

## Recent experiment ledger (this session)

- `enyo-1.33.0-rc9`: self-play generated with the wrong policy net
  (Stockfish's own net loaded into the search engine instead of the
  candidate's), giving off-policy data with no self-distillation signal.
  Rejected, elo=-17.3 vs rc10 at 1708/6000 games.
- `enyo-1.33.0-rc10` (fine-tune): corrected the rc9 mistake - candidate
  self-play, genuine static-eval relabel, matching the actual rc57 recipe.
  Still rejected, elo=-9.0 at 3014/6000 games. The incremental
  self-play-fine-tune path off rc10 appears saturated.
- `enyo-1.33.0-rc11` (scratch round 3, continuing the historical
  rc3->rc4 scratch bootstrap on the corpus expanded with self-play data):
  paused mid-run at superbatch 38217/83770 (46%) to prioritize a
  higher-conviction experiment. Checkpoint preserved at
  `runs/enyo-1.33.0-rc11/checkpoints/native-38000` if resuming is ever
  worthwhile; extrapolating rounds 1->2's own decay (-460 -> -213.8 ->
  -180.6) suggests this round plateaus around -150 to -165, still short
  of rc10 - not a strong case for resuming as-is.
- `enyo-1.33.0-rc12`: tested output_bucket_scope=full-head (independent
  per-output-bucket L1/L2 weights, matching every reference engine
  checked - Stockfish, PlentyChess, Reckless all use this instead of this
  lineage's long-standing shared-head default) built on rc10 via a
  weight-replication warm start. Rejected, elo=-5.2 at 5620/6000 games -
  real, but only 1 fine-tune epoch, likely not enough for the 8
  initially-identical head copies to differentiate.
- `enyo-scc-1.0.0-rc1` (new lineage: genuine random-init scratch train on
  the completed Lc0 conversion combined with nodes5000pv2, 3,232,772,104
  positions, properly interleaved via bullet-utils interleave since the
  `direct` loader has no shuffle buffer of its own (just a 256MB
  read-ahead one), 9 epochs, wdl=0.05 - the scratch-long precedent
  regimen, vs champion enyo-1.32.0-rc10 directly since there is no
  continue_from): rejected, elo=-93.2 llr=-32.60/690.78 (-5%) at
  4000/4000 games - expected for a single-generation scratch lineage
  against a champion many fine-tune iterations deep (enyo-1.0.0-rc1's own
  founding net started at -181.2 vs SF). Not evidence against the Lc0
  data - the champion comparison decides promotion, not data quality.
- `enyo-scc-1.1.0-rc1` (dose-continuation round 1: continue_from rc1.0,
  same regimen/corpus + one-time existing self-play blend, 231321
  superbatches): rejected, elo=-66.9 llr=-26.24/690.78 (-4%) at 4000/4000
  games - +26.3 Elo over rc1.0 on the same reference, validating
  dose-continuation as the ongoing lever (matches enyo-scratch-long's
  early-round gains).
- `enyo-scc-1.1.0-rc2` (shuffle retry of rc1.1, same parent rc1.0, same
  regimen): residual gate rejection, never reached SPRT. endgame
  slope_gain=-0.004 and eval:800+ slope_gain=-0.001, both just below
  MIN_SLOPE=0.05 threshold. MAE improved (endgame 313.8→227.8) but
  calibration vs champion did not. Margin of failure is within noise; shuffle
  hypothesis inconclusive.
- `enyo-scc-1.2.0-rc1` (continue_from rc1.1, WDL 0.05→0.3 + cubic loss
  |e|^3 + AdamW beta1=0.95 + final_lr 5e-6→2.5e-5): rejected,
  vs-SF: elo=-223.9 ci=13.9 (1500 games) - catastrophic, 87 Elo worse than
  the founding net. Most likely cause: WDL=0.3 conflicts with Lc0 MCTS game
  results vs SF oracle eval labels in 38% of corpus. Cubic loss and beta1
  are not the culprit (cannot account for this magnitude of regression).
- `enyo-fullhead-threats-v1-rc1`: full-head combined with FullThreats
  (format v6, added this session across the enyo C++ loader, Rust
  trainer, and Python export library - was previously blocked by a
  deliberate one-variable-at-a-time guard, not a technical limitation).
  Warm-started from rc10, properly dosed (20 epochs). Rejected on
  residual gate (mae improves but slope_gain goes negative in
  endgame/high-eval, the same distortion signature as TB-corrected
  labels), then confirmed by a direct SPRT: elo=-119.2 at 250/6000 games,
  real LLR-bound crossing. This closes both full-head and FullThreats as
  live leads for now - full-head got a real, adequately-dosed test this
  time (unlike rc12 alone) and still failed.
- FullThreats/slider-xray in isolation: already closed before this
  session, extensively tested with real dose (up to 32,768 superbatches
  across multiple configs), never recovered competitive strength. Do not
  reopen without a genuinely new variable.

## enyo-lc lineage (2026-08-07–08)

New lineage using exclusively lc0 nodes5000pv2 self-play data
(`data/bullet/combined/lc0-nodes5000pv2-selfplay-1.2-shuffled.bullet`).
Plain ReLU, scratch start, cubic loss, AdamW beta1=0.95, final_lr=2.5e-5.

- `enyo-lc-1.0.0-rc1 (WDL=0.3)`: abandoned at 436/4000 SPRT games.
  SF gate absolute: elo≈-328 vs SF oracle (benchmarks/stockfish-net.jsonl,
  3000 games), implying ≈-188 vs rc10. Root cause: 53% of lc0 self-play
  positions within 100cp of 0; at WDL=0.3 the noisy MCTS game outcome floods
  training for balanced positions. Same mechanism as enyo-scc-1.2.0-rc1.
- `enyo-scc-1.2.0-rc2` (dose round 2, continue_from scc-1.1.0-rc1, WDL=0.05, cubic
  loss, beta1=0.95, final_lr=2.5e-5): **FAILED residual gate**. MAE improved
  (endgame +16.5cp, eval:800+ +30.9cp) but slope_gain was -0.028/-0.034 for those
  groups (< MIN_SLOPE=0.05). Root cause: cubic loss produces more conservative
  eval magnitudes for extreme positions than squared loss, causing slope degradation
  vs a squared-loss reference net. The gate was a false rejection — actual playing
  quality unknown. Net exists at assets/nets/enyo-scc-1.2.0-rc2.nn.
- `enyo-scc-1.3.0-rc1` (WDL=0.3 retry via initialize_from scc-1.2.0-rc1):
  INTERRUPTED at superbatch 4. Abandoned.
- `enyo-scc-2.0.0-rc1` (SCReLU via initialize_from scc-1.2.0-rc2): Passed static
  eval gate (slope=0.869). No residual gate (initialize_from, not continue_from).
  SF gate did not complete (run interrupted). Net exists at assets/nets/enyo-scc-2.0.0-rc1.nn.
- `enyo-scc-2.0.0-rc2` (SCReLU, same config, skip_stockfish_gate=true): INTERRUPTED
  at superbatch 104847/231321 (~45%). Session switched to lc0 WDL ablation.
- `enyo-lc-1.0.0-rc1 WDL=0.075 (pwa-llm)`: SF gate absolute elo=-274.7 ci=9.4
  (4000 games). Gate vetoed — delta=-131.3 vs rc10 at -143.4, upper90=-123.4 < 0.
  Never reached SPRT. This is the primary comparator for the WDL ablation.
- `enyo-lc-1.0.0-rc1 WDL=0.05 (pwa-5090, enyo-tmp.nn)`: SPRT running directly
  via sprt_net.py after manual export (existing file had to be removed first).
  SF gate absolute Elo pending. Winner vs 0.075 by less-negative absolute Elo;
  if within CI (25 Elo), default to WDL=0.05.

**Residual gate fix (2026-08-08)**: Added `skip_residual_gate` build.json field
(nnue script + defaults.json). All dose continuation runs with cubic loss must
include `"skip_residual_gate": true` to bypass the slope-calibration false rejection.
SF gate + SPRT remain as real quality arbiters.

## Open questions / in flight

- **WDL for lc0 self-play data**: ablation in progress (0.05 vs 0.075). The
  scc lineage used WDL=0.05 for its founding net (worked), WDL=0.3 failed
  catastrophically on both scc and lc0 self-play data. Correct value is ≤0.1;
  0.05 vs 0.075 ablation will narrow this down.
- **Dose-continuation (lc0 lineage)**: once the WDL ablation winner is known,
  run enyo-lc-1.1.0-rc1 with continue_from=winning_net, same WDL, same data.
  Every dose build.json MUST have ALL THREE of:
  1. `"skip_residual_gate": true` — cubic loss causes slope_gain false rejection
  2. `"skip_stockfish_gate": true` — SF gate MIN_DELTA=0 vetoes any net weaker
     than rc10 in absolute Elo; dose rounds start at ~-250 absolute and gain
     ~25 Elo/round, so the gate blocks progress for many rounds. SPRT vs rc10
     is the real arbiter; the SF gate was only useful for measuring the founding
     net's absolute Elo (already done: -274.7 for WDL=0.075).
  3. `"reference": "enyo-1.32.0-rc10"` — without this, the nnue script falls
     back to continue_from as the SPRT reference; the dose net compares against
     its own weak parent and falsely passes SPRT.
  Expected gain: +20-30 Elo per round (scc round 1 measured +26.3 Elo;
  diminishing returns apply in later rounds). Repeat until a net passes SPRT
  vs rc10.
- **SCReLU via initialize_from**: BLOCKED until the lc0 lineage produces an
  **accepted** (SPRT-passing) net. When unblocked: architecture.json adds
  `dense_activation=relu-screlu-residual` + changes `export_format=enyo-native-v4`;
  build.json adds `initialize_from=<accepted_lc0_net>`.
  **All prior SCReLU tests were invalid**: scc-2.0.0-rc1/rc2 ran with l2sb gradient
  frozen (FusePointwise IR compiler bug in bullet_lib; fixed in commit 81d319de via
  bullet-patched). The squared branch bias never updated. Future SCReLU tests have
  both fixes: correct WDL (0.05–0.075) and working l2sb gradient. Expected: 10–25 Elo.
- Lc0 T91 bulk data (scc corpus, 3.23B positions): fully converted, tested
  as enyo-scc lineage. Founding net at -93.2 vs rc10, dose round 1 at -66.9.
  enyo-scc is paused; lc0 self-play (pure self-play, fresher data) is the
  active lineage to compare. If lc0 self-play founding net and subsequent
  rounds outperform scc's trajectory, scc may be retired.

## Post-SCReLU architectural leads (not yet tested)

In priority order. Each requires a separate architecture number and engine
support. Do not combine with data or LR changes.

- **Skip connection L2→L3** (medium confidence, ~10-20 Elo estimated): Add a
  direct residual path from the 16-neuron L2 output to the output bucket,
  bypassing the final linear transform. Used by Stormphrax and PlentyChess.
  Enyo has no skip connection anywhere in the dense head. Requires new export
  format (enyo-native-v5 or similar) and engine loader change. Test via
  `initialize_from` on an accepted SCReLU net.

- **L2 size 16→32** (low-medium confidence): Enyo uses l2_size=16 (dense
  intermediate layer). Stormphrax uses 32. Doubling this adds capacity without
  major architecture upheaval. Requires export format change. Pairs naturally
  with skip-connection work since both touch the dense head. Test after
  skip-connection result is known.

- **FullThreats clean from-scratch test** (high variance, 0–100 Elo):
  The only FullThreats test combined it with full-head (two variables) and
  warm-started from a non-threats net (threat weights zero-initialized from a
  net with no threat knowledge). Result was -119 Elo. Full-head alone was -5
  Elo, so threats were the culprit — but initialization and variable conflation
  mean the test was not clean. A true clean test requires scratch training with
  FullThreats alone, correct WDL, from random init. Format v6 (FullThreats)
  is already merged. High-risk: could fail again, or could add 50+ Elo.
  Do not reopen until WDL and SCReLU are settled.

- **Input buckets 16→32** (low confidence): Stockfish uses 32. Finer
  king-position grouping might help at top strength. Requires architecture
  + engine changes, expensive to test. Low priority until above are exhausted.

## Rules

Follow `~/.claude/skills/nnue-*/SKILL.md` (also mirrored at
`~/.codex/skills/nnue-*/SKILL.md`) for the authoritative iteration policy,
coding, validation, and git roles. Read them before starting new work in
this repository - do not rely on this document for procedural rules, only
for current experiment status.
