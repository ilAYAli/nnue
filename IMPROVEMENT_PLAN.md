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
  positions, 9 epochs, wdl=0.05 - the scratch-long precedent regimen,
  vs champion enyo-1.32.0-rc10 directly since there is no continue_from):
  rejected, elo=-110.8 llr=-33.60/690.78 (-5%) at 4000/4000 games - a
  severe, not marginal, loss. Root cause identified: the combined corpus
  was built by flat `cat`-concatenating lc0-static-bulk.bullet
  (1,232,772,104 positions) then nodes5000pv2-recalibrated-2b.bullet
  (2,000,000,000 positions) with no shuffling, and the `direct` loader
  (`bullet_lib::value::loader::direct::DirectSequentialDataLoader`, read
  directly from the vendored crate source) reads file(s) in strict
  sequential order with only a 256MB read-ahead buffer - not a shuffle
  buffer. The two very differently-distributed corpora were therefore
  never blended within a batch: every one of the 9 epochs cycled through
  one long unbroken run of pure lc0 positions (~38% of an epoch) followed
  by one long unbroken run of pure nodes5000pv2 positions (~62%),
  identically each pass. Any future combined-corpus attempt must shuffle
  or interleave at the position level before training, not just
  concatenate files.
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

## Open questions / in flight

- Lc0 bulk data: fully downloaded and converted via a distributed Forge
  to-bullet --label static job (30,174 shards, label-static-test91-20260803-080521,
  1.906B raw records -> 1,232,772,104 selected positions after
  quiet-only/min-ply=16 filtering), then combined with
  nodes5000pv2-recalibrated-2b.bullet into a 3.23B-position corpus and
  tested as enyo-scc-1.0.0-rc1 (see ledger) - rejected, but for a
  data-preparation reason (unshuffled concatenation feeding the
  sequential-only `direct` loader), not because the Lc0 volume itself is
  bad. Re-testing with a properly shuffled/interleaved combined corpus is
  the natural next step before drawing any conclusion about the Lc0 data
  actual value.
- No architecture or data lever currently has a strong, well-justified
  case for reopening. The next real experiment should come from the Lc0
  data volume once its conversion pipeline exists, or from a fresh
  hypothesis grounded in comparison against the reference engines in
  `~/source/` (Stockfish, Obsidian, Alexandria, PlentyChess, Reckless,
  Stormphrax, berserk, pawnocchio, rice, viridithas, ShashChess).

## Rules

Follow `~/.claude/skills/nnue-*/SKILL.md` (also mirrored at
`~/.codex/skills/nnue-*/SKILL.md`) for the authoritative iteration policy,
coding, validation, and git roles. Read them before starting new work in
this repository - do not rely on this document for procedural rules, only
for current experiment status.
