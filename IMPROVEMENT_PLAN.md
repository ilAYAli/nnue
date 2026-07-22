# NNUE Architecture Improvement Plan

## Current status

Current: **Scale-root active tests**

- Absolute benchmark target: Stockfish net `nn-0ee0657fb25e.nnue`, not
  `default.net`.
- Current promoted Enyo family is around `-150` to `-180` Elo versus the
  Stockfish net and about `-140` Elo versus the Berserk net in short fixed
  checks. Twenty-ish same-lineage continuation/fine-tune attempts have not
  closed the gap.
- Do not spend the next iteration on "more of the same" training or another
  Stockfish/Reckless-inspired feature. The near-term target is a structural
  explanation for why an architecture close to Berserk is still far weaker.
- FullThreats support was implemented and tested on 2026-07-09:
  - Rust trainer can emit `full_threats=true` feature rows.
  - Enyo native v2 header uses a strict FullThreats flag.
  - Enyo runtime loads, reports, and searches with flagged FullThreats nets.
  - FullThreats currently requires `export_format=enyo-native-v2` and
    `input_factoriser=false`.
- Verification completed:
  - `cargo check --features cuda --bin train`
  - `cargo check --features cuda`
  - `cargo test --features cuda --bin train`
  - Enyo build
  - one-superbatch FullThreats export/load/search smoke
- Known unrelated test issue: full Enyo test suite currently has one failing
  search expectation, `search.hypersion_check_net_root_evasions_find_safer_defenses`.
  Network/model/audit tests passed.
- Structural audit tools added:
  - `tools/validate/structural_net_audit.py`
  - `tools/validate/fit_static_eval_scale.py`
  - `tools/validate/native_activation_audit.py`
  - `tools/validate/scale_output_head.py`
- First static audit on 150 fixed FENs extracted from Enyo tests/scripts:
  - `enyo-1.28.0-rc16.nn` vs `berserk-9b84c340af7e.nn`:
    `mae=429.41`, `rmse=741.08`, `corr=0.8305`, `slope=0.5952`,
    `sign_disagree=17/150`.
  - Enyo saturated/clipped much more often than Berserk:
    `abs>=2000` was `48/150` for Enyo versus `18/150` for Berserk.
  - The largest grouped disagreement was in endgames:
    `mae=559.86`, `corr=0.7511`, `slope=0.4622`.
  - The same run against Stockfish showed similar broad scale/distribution
    trouble, so the issue is not opponent choice.
- Follow-up audit on 500 positions from
  `/home/petter/assets/enyo_nnue_data/sf_labels/sf_combined_904k.val.jsonl`:
  - Enyo was much hotter than both references:
    - Berserk fit: `scale=0.5246`, `bias=-16.42`,
      `mae 206.30 -> 88.34`, `corr=0.9100`.
    - Stockfish fit: `scale=0.4971`, `bias=-2.11`,
      `mae 219.19 -> 81.43`, `corr=0.8956`.
  - Final score clamp frequency was `14/500` for Enyo versus `1/500` for both
    Berserk and Stockfish.
  - The native activation audit on the same 500 positions found no input CReLU
    high-cap saturation (`incap=0.00%`), but final outputs are too wide:
    `raw_sd=654.65`, `scaled_sd=762.69`, `clamp=14/500`.
  - Output buckets 0-5 carry the pathological variance/clamps, while buckets
    6-7 are comparatively tame. Endgames remain the largest material-phase
    problem: `raw_sd=1308.64`, `scaled_sd=1434.44`, `clamp=7/53`.
  - Interpretation: this is primarily final-score calibration/output-head
    distribution trouble, not a missing FullThreats-style input feature. A
    global post-scale would recover a large amount of static agreement but is
    not sufficient by itself, especially in endgames.
- Export bug fixed for the completed PSQT residual candidate:
  `tools/bullet/spike_trainer/src/bin/train.rs` now includes the v5 PSQT
  residual table in checkpoint payload sizing, so valid PSQT tensors are no
  longer rejected as an unexpected trailer. Regression:
  `cargo test --manifest-path tools/bullet/spike_trainer/Cargo.toml --features cuda --bin train`.
- `enyo-5.0.0-rc1` exported successfully to
  `/home/petter/assets/nets/enyo-5.0.0-rc1.nn` and passed startpos plus
  50k-row static validation (`mae=260.480`, `corr=0.766462`,
  `slope=0.336371`). SPRT result:
  `sprt-enyo-5.0.0-rc1.nn-vs-enyo-1.20.0-rc12.nn-1500-20260713-233949`
  scored `-7.9 +/-14.3` Elo with `14.0%` LOS and `36.1%` draws. Reject
  PSQT residual as currently trained; do not spend another lane on PSQT before
  the output-scale issue is resolved.
- Fast scale-root candidate exported by scaling only the current champion final
  output head by `0.52`:
  `/home/petter/assets/nets/enyo-1.30.0-rc1.nn`.
  On the 500-position validation audit, compared with the unscaled champion:
  - clamp rate dropped from `14/500` to `3/500`;
  - Berserk MAE dropped to `91.02`;
  - Stockfish MAE dropped to `85.48`.
  SPRT result:
  `sprt-enyo-1.30.0-rc1.nn-vs-enyo-1.28.0-rc16.nn-1500-20260713-234121`
  scored `+14.8 +/-14.1` Elo with `98.0%` LOS and `37.6%` draws. Absolute
  Berserk check failed:
  `sprt-enyo-1.30.0-rc1.nn-vs-berserk-9b84c340af7e.nn-1500-20260714-013616`
  was stopped at `-128.9` Elo, `llr=-2.38/2.20`, after 358 games.
- Scale grid candidates exported:
  - `/home/petter/assets/nets/enyo-1.30.0-rc6.nn`
  - `/home/petter/assets/nets/enyo-1.30.0-rc5.nn`
  - `/home/petter/assets/nets/enyo-1.30.0-rc3.nn`
  - `/home/petter/assets/nets/enyo-1.30.0-rc4.nn`
  - `/home/petter/assets/nets/enyo-1.30.0-rc2.nn`
  Static audit on the same 500 validation positions suggests `0.48` is slightly
  better than `0.52`: Berserk MAE `90.26`, Stockfish MAE `82.17`. `0.44` is
  better on Stockfish static MAE (`81.54`) but worse on Berserk static MAE
  (`92.35`), so it is queued as a follow-up game test rather than promoted.
  SPRT result:
  `sprt-enyo-1.30.0-rc3.nn-vs-enyo-1.28.0-rc16.nn-1500-20260714-003933`
  scored `+25.3 +/-13.9` Elo with `100.0%` LOS and `39.3%` draws. It is the
  current best global scale. Absolute Berserk check failed:
  `sprt-enyo-1.30.0-rc3.nn-vs-berserk-9b84c340af7e.nn-1500-20260714-022347`.
  It scored `-112.8` Elo, `llr=-2.24/2.20`, after 392 games. This recovers
  about 30 Elo against Berserk versus the original short `-142.9` result, but
  leaves most of the gap. Paused follow-up:
  `sprt-enyo-1.30.0-rc6.nn-vs-enyo-1.28.0-rc16.nn-1500-20260714-014410`
  at `+26.1` Elo after 40 games.
- Output-only calibration candidate trained from the `0.52` scale-root net:
  `/home/petter/assets/nets/enyo-1.31.0-rc1.nn`.
  It passed startpos and 50k-row static validation (`mae=260.186`,
  `corr=0.766556`, `slope=0.335897`). Static shape is nearly unchanged versus
  the scale-root parent. SPRT result:
  `sprt-enyo-1.31.0-rc1.nn-vs-enyo-1.30.0-rc1.nn-1500-20260714-003855`
  scored `-22.0 +/-13.8` Elo with `0.1%` LOS and `40.5%` draws. Reject
  output-only calibration; keep pure final-head scaling as the parent lane.
  `tools/validate/scale_followup.sh` now guards against duplicate Berserk
  launches when tmux normalizes dotted session names.
- `tools/validate/scale_output_head.py` now accepts per-output-bucket
  `--bucket-scales`. Averaging Berserk and Stockfish 500-row per-bucket
  through-origin fits produced:
  `0.340267,0.548593,0.560764,0.452566,0.495903,0.586191,0.514314,0.494798`.
  Exported candidate:
  `/home/petter/assets/nets/enyo-1.31.0-rc2.nn`.
  On the 500-position static audit it scores Berserk MAE `88.58` and Stockfish
  MAE `83.54`, so it improves Berserk agreement versus global `0.48` but is
  slightly worse versus Stockfish. Reference SPRT result:
  `sprt-enyo-1.31.0-rc2.nn-vs-enyo-1.28.0-rc16.nn-1500-20260714-014953`
  scored `+11.8 +/-13.9` Elo with `95.3%` LOS and `39.5%` draws. Absolute
  Berserk SPRT failed:
  `sprt-enyo-1.31.0-rc2.nn-vs-berserk-9b84c340af7e.nn-1500-20260714-033829`.
  It scored `-117.4` Elo, `llr=-2.29/2.20`, after 390 games, so static
  per-bucket scaling did not improve the absolute Berserk gap versus global
  `0.48`.
- Trained bucket-output calibration candidate:
  `/home/petter/assets/nets/enyo-1.31.0-rc3.nn`.
  It initializes from the static bucket-average net, trains only `output`, and
  uses a reduced dose/LR (`96` superbatches, `lr=0.00001`) to avoid destroying
  the fixed dense/input path. Gates passed: startpos `+69cp`, static
  `mae=261.096`, `corr=0.765900`, `slope=0.336176`. SPRT result:
  `sprt-enyo-1.31.0-rc3.nn-vs-enyo-1.28.0-rc16.nn-1500-20260714-034233`
  scored `+0.2 +/-14.4` Elo with `51.3%` LOS and `34.9%` draws. Reject as
  neutral.
- Trained float-head candidate:
  `/home/petter/assets/nets/enyo-1.31.0-rc4.nn`.
  It initializes from the best global-scale parent `0.48`, trains `float-head`
  (`L2/L3/output`) for `128` superbatches at `lr=0.00001`, and leaves the input
  layer plus int8 L1 fixed. Gates passed: startpos `+49cp`, static
  `mae=261.351`, `corr=0.766261`, `slope=0.335136`. SPRT result:
  `sprt-enyo-1.31.0-rc4.nn-vs-enyo-1.28.0-rc16.nn-1500-20260714-035544`
  scored `-3.5 +/-14.2` Elo with `31.5%` LOS and `37.0%` draws. Reject.
- Trained dense-head candidate:
  `/home/petter/assets/nets/enyo-1.31.0-rc5.nn`.
  It initializes from the best global-scale parent `0.48`, trains `dense-head`
  (`L1/L2/L3/output`) for `96` superbatches at `lr=0.000005`, and leaves the
  input layer fixed. Gates passed: startpos `+51cp`, static `mae=261.450`,
  `corr=0.765547`, `slope=0.333147`. SPRT result:
  `sprt-enyo-1.31.0-rc5.nn-vs-enyo-1.28.0-rc16.nn-1500-20260714-041508`
  scored `+0.0 +/-14.3` Elo with `50.0%` LOS and `35.6%` draws. Reject as neutral.
- Trained input-only candidate:
  `/home/petter/assets/nets/enyo-1.31.0-rc6.nn`.
  It initializes from the best global-scale parent `0.48`, trains only input
  embeddings for `128` superbatches at `lr=0.000005`, and keeps dense/output
  layers fixed. Gates passed: startpos `+21cp`, static `mae=292.542`,
  `corr=0.784237`, `slope=0.189592`. Reference SPRT result:
  `sprt-enyo-1.31.0-rc6.nn-vs-enyo-1.28.0-rc16.nn-1500-20260714-043816`
  scored `+22.3 +/-13.6` Elo with `99.9%` LOS and `41.6%` draws. Absolute
  Berserk SPRT is active:
  `sprt-enyo-1.31.0-rc6.nn-vs-berserk-9b84c340af7e.nn-1500-20260714-063318`.
- Attempted broad-data input-only candidate
  `/home/petter/assets/nets/enyo-1.31.0-rc7.nn` using
  `data/bullet/enyo-scratch-broad-1.0.0-rc1.bullet`; it exported and passed
  gates, but is byte-identical to `enyo-1.31.0-rc6.nn`
  (`sha256=8a3a3663...`). Do not run a duplicate SPRT. Before spending more
  "different data" attempts, verify the direct Bullet loader/build path is
  actually changing the sampled rows.

Next:

1. Let the active input-only Berserk SPRT reach a decision.
2. If it still loses badly to Berserk, audit data materialization/sampling and prepare the next non-duplicate input/data lane.
3. Test only candidates that beat the champion convincingly against Berserk;
   otherwise audit data materialization/sampling before another data-regime
   attempt.
4. Treat FullThreats/architecture expansion as secondary until the output-scale
   pathology is resolved.

## Unified scratch architecture and feature competition

This competition supersedes the earlier architecture freeze and the staged
architecture-screening proposal for this one purpose. It is a single fixed
competition: every eligible configuration is trained from random initialization
to the full dose, no candidate is eliminated early, and all selection rules are
declared before the first run. The result identifies the best tested
architecture/feature combination for Enyo under the fixed corpus, objective,
training budget, quantized runtime, and game protocol. It does not claim that an
untested architecture or a different training regime cannot be stronger.

### Competition field

All Enyo-mode candidates use `l2_size=16`, `eval_scale=400`, `l0_std=8`,
`l1_std=1`, `l1_export_scale=1`, and the current native container format unless
the row explicitly changes a field. Every candidate starts from scratch using
the same deterministic initialization method and distributions; `rc1` uses seed
`5090001` and `rc2` uses seed `5090002` for every architecture. Different tensor
shapes cannot contain byte-identical weights, but no candidate receives a
pretrained or otherwise privileged initialization. `oN` means
`output_buckets=N`.

The lineage slug is the stable architecture identity used in artifact names:
`enyo-<lineage>-v1-rc1` and `rc2` are the two matched scratch seeds.

| ID | Lineage | Configuration | Status and purpose |
| --- | --- | --- | --- |
| A | `h1` | `1x12-1024-o8`, unfactorised | Completed. No-HalfKA/king-independent control. |
| B | `h4` | `4x12-1024-o8`, factorised | Completed. Light king conditioning with shared factoriser. |
| C | `h8` | `8x12-1024-o8`, factorised | Completed. Medium king conditioning and the historically positive eight-bucket idea. |
| D | `h16` | `16x12-1024-o8`, factorised | Completed. Current Enyo architecture control. |
| E | `h10w768` | `10x11-768-o8`, factorised | Ineligible under the current runtime: the 11-channel HalfKAv2 mapping requires 32 input buckets. |
| F | `sf32` | `32x11-1024-o8`, factorised | Completed. High-capacity, data-hungry Stockfish-like HalfKAv2 extreme. |
| G | `h16w768` | `16x12-768-o8`, factorised | Completed. Width-only challenger. |
| H | `h16o4` | `16x12-1024-o4`, factorised | Completed. Output-bucket challenger. |
| I | `h10w768o4` | `10x11-768-o4`, factorised | Ineligible under the current 11-channel input-bucket contract. |
| J | `recklessft` | current Reckless `10x12-768-o8`: native mirrored king buckets, 66,864 occupied-piece threat features, factorised piece inputs, pairwise CReLU, explicit eight-bucket material map, and full dense heads | Rejected. After fixing the dense-export layout and re-exporting the preserved checkpoint, rc1 scored `1-1486-13` (`-919.54 +/-93.68` Elo, `0.0%` LOS) versus `enyo-1.30.0-rc3`; do not train rc2. The earlier broken export and piece-only run are invalid provenance, not architecture results. |
| K | `h10w768u` | `10x11-768-o8`, unfactorised | Proposed matched control for FullThreats; requires a compatible 11-channel input-bucket contract. |
| L | `h10w768ft` | `10x11-768-o8`, unfactorised, FullThreats | Proposed tactical-feature challenger; requires the same compatible contract as K. |

The candidate letter is only a compact table reference; use the lineage slug in
run names, results, and discussion. Candidate J follows current Reckless rather
than the abandoned piece-only network: its piece bucket map, threat indexing,
output bucket boundaries, pairwise activation, quantisation, and input-major
dense layout must all match between trainer and Enyo. It becomes eligible only
after an exported checkpoint passes feature and score parity plus the fixed NPS
gate. Candidate L must be compared directly
with K because FullThreats does not support the input factoriser. Its final game
result includes its runtime cost; historical FullThreats NPS loss must not be
ignored.

### Fixed training method

Do not use `enyo-scratch-long-1.0.0-rc1` alone as the training recipe. Its
`+93.2 Elo` was parent-relative to a weak broad baseline, while its first
absolute default-net result was still about `-169 Elo`. It was the start of the
successful lineage, not a sufficiently trained endpoint.

The accepted lineage eventually used five `196608`-superbatch passes followed
by two `98304`-superbatch passes, but reproducing that entire trajectory for
every architecture and seed would require roughly 550 GPU-hours. That protocol
was stopped after candidate A's first pass when its cost became clear.

The competition instead uses one uniform `196608`-superbatch scratch training
per architecture and seed. This is the longest historically demonstrated
single-pass recipe, keeps the LR schedule identical across shapes, and makes the
race practical without staged training. With twelve configurations and two
seeds, the fixed field contains 24 trainings and is expected to require about 62
GPU-hours on the measured host. The result selects the architecture that
performs best at this fixed compute budget; deeper winner training is separate.

Use the canonical source directly:
`data/stockfish/master-binpacks/training_data_pylon.binpack`, offset `0`, limit
`2800000000`, with `sfbinpack.min_ply=24`. Do not depend on the deleted
`data/bullet/enyo-scratch-broad-1.0.0-rc1.bullet` intermediate. Record the
source SHA-256, convert the declared slice once, and reuse that immutable Bullet
artifact so all architectures consume identical accepted rows in identical
order. Do not independently convert, resample, or relabel the source per
architecture.

The canonical `build.json` shape is:

```json
{
  "run": "enyo-A.P.0-rcN",
  "hypothesis": "competition replicate: random-init ARCH under the fixed one-pass pylon protocol",
  "superbatches": 196608,
  "init_seed": 5090001,
  "wdl": 0.05,
  "data": {
    "source_binpack": "data/bullet/enyo-architecture-race-pylon-2.8b.bullet"
  }
}
```

Every run omits both origins and starts from scratch. `lr`, `final_lr`, batches,
batch size, loader,
trainable scope, weight decay, and threads stay inherited from `defaults.json`;
before kickoff, verify they still resolve respectively to `0.001`, `0.000005`,
`64`, `2048`, `direct`, `all`, `0.0`, and `16`.

The later accepted lineage gained further Elo from WDL `0.15`, activation L1,
lower learning rates, and disjoint Farseer/T60T70 slices. Those are valuable
post-foundation optimization evidence but are not folded into this competition:
doing so would test a long sequence of objective and corpus interactions rather
than architecture/feature capacity. The race selects the best architecture
under a common fixed compute budget; it does not claim to finish all subsequent
net tuning.

### Matched continuation competition

The scratch scores are foundation measurements, not the final architecture
ranking. Four lineages form the credible cluster and advance: `h4`, `h8`,
`h16`, and `sf32`. Both scratch seeds advance for every lineage; selecting only
the strongest seed would bias the comparison. The weaker `h1`, `h16w768`, and
`h16o4` controls do not advance, and the catastrophic corrected `recklessft`
result rejects that lineage.

Each of the eight descendants uses its exact scratch net as `continue_from` and
receives the same historically successful continuation treatment: 256
superbatches from `T60T70wIsRightFarseer.binpack` at offset 0 and limit
200,000,000, `lr=0.0001`, `wdl=0.05`, and `activation_l1=0.00001`. Architecture
and all other resolved settings remain unchanged. First test each descendant
against its own ancestor to measure refinement gain, then against immutable
`enyo-1.30.0-rc3.nn` to measure absolute strength. Rank lineage trajectories by
the paired two-seed evidence and directly test the leading refined lineages;
never declare an architecture winner from the best isolated scratch RC.

First continuation result: `enyo-h4-v2-rc1` gained `+21.3 +/-14.5` Elo over
its own `enyo-h4-v1-rc1` ancestor in 1,500 games (`0.93/2.20` LLR, `99.8%`
LOS, `39.2%` draws). Continue the matched field; do not select `h4` from one
seed. Direct mature-reference matches are deferred until the leading refined
lineages are known.

The second `h4` seed gained `+8.8 +/-14.9` Elo (`0.38/2.20` LLR, `87.7%`
LOS, `36.7%` draws) over its own ancestor. The approximate two-seed mean is
`+15.1 +/-10.4` Elo. This establishes a replicated positive `h4` refinement
response; continue with the same two-seed protocol for `h8`, `h16`, and
`sf32`.

The first `h8` continuation seed gained `+22.5 +/-14.3` Elo (`1.04/2.20`
LLR, `99.9%` LOS, `40.5%` draws) over `enyo-h8-v1-rc1`. Run the matched second
seed before comparing the `h8` trajectory with the completed `h4` mean.

The second `h8` seed gained `+10.7 +/-14.8` Elo (`0.47/2.20` LLR, `92.0%`
LOS, `36.9%` draws). The approximate `h8` two-seed mean is `+16.6 +/-10.3`
Elo, indistinguishable from `h4` at this depth. Continue with `h16` and
`sf32`; do not break the tie using isolated RCs.

The first `h16` continuation seed gained `+25.3 +/-14.7` Elo (`1.15/2.20`
LLR, `100.0%` LOS, `38.6%` draws) over `enyo-h16-v1-rc1`. This is the largest
first-seed gain so far but overlaps the other lineages; run `h16` seed 2 before
comparison.

The second `h16` seed gained `+20.6 +/-14.5` Elo (`1.01/2.20` LLR, `99.7%`
LOS, `39.7%` draws). The approximate `h16` two-seed mean is `+23.0 +/-10.3`
Elo. It leads numerically but is not significantly separated from `h4` or
`h8`; complete both `sf32` seeds before deciding the next depth.

The first `sf32` continuation seed gained `+15.3 +/-14.7` Elo (`0.67/2.20`
LLR, `97.9%` LOS, `38.4%` draws) over `enyo-sf32-v1-rc1`. Complete the matched
second seed before comparing lineage means or selecting the next training dose.

The second `sf32` seed gained `+14.8 +/-14.4` Elo (`0.71/2.20` LLR, `97.8%`
LOS, `40.3%` draws). The approximate two-seed `sf32` mean is `+15.1 +/-10.3`
Elo. At the 256-superbatch checkpoint, the four lineage means are `h4 +15.1`,
`h8 +16.6`, `h16 +23.0`, and `sf32 +15.1` Elo, all with approximately
`+/-10.3` to `+/-10.4` uncertainty. No pair is significantly separated.

Continue all four lineages and both seeds rather than selecting the numerical
leader. Each `v3` descendant continues its exact `v2` seed for 7,600
superbatches with the same `lr=0.0001`, `wdl=0.05`, and
`activation_l1=0.00001` objective. Use the next disjoint
`T60T70wIsRightFarseer.binpack` range (`offset=200,000,000`,
`limit=1,000,000,000`), which covers the 996,147,200 requested positions
without wrapping the short checkpoint's data. Test each descendant against its
own `v2` parent for at most 1,500 games. This measures whether architecture
learning trajectories diverge with materially more training; mature-reference
and direct finalist matches remain deferred until all eight long continuations
are complete.

The first long `h4` continuation scored `+4.2 +/-14.8` Elo (`0.20/2.20` LLR,
`71.0%` LOS, `37.6%` draws) over its `v2` parent. This is inconclusive but a
much weaker marginal gain than its 256-superbatch checkpoint. Run the first
long seed of `h8`, `h16`, and `sf32` next; only then decide which lineages merit
the second long seed, avoiding another eight-run commitment when the learning
curves may already separate.

The first long `h8` continuation failed the fixed residual gate: endgame MAE
improved `420.366 -> 412.136` and eval 800+ improved `643.374 -> 625.768`, but
eval 300-799 regressed `350.209 -> 350.921`. Do not run its SPRT. Continue with
the first long `h16` and `sf32` seeds before making the candidate cut.

The first long `h16` continuation also failed the residual gate: endgame MAE
improved `409.651 -> 405.158` and eval 800+ improved `626.343 -> 616.099`, but
eval 300-799 regressed `352.789 -> 354.122`. Do not run its SPRT. The repeated
failure in the same band means the 7,600-superbatch regimen cannot currently be
used to eliminate `h8` relative to the incumbent. Run `sf32` seed 1, then
reassess the common continuation dose/objective before any second seeds.

The first long `sf32` continuation likewise failed: endgame MAE improved
`415.116 -> 406.494` and eval 800+ improved `634.993 -> 614.823`, while eval
300-799 regressed `347.194 -> 352.502`. Do not run its parent-relative SPRT or
any second long seeds. Since `h8`, `h16`, and `sf32` all failed the same band,
the long regimen is not a valid discriminator. Use direct head-to-head games
between the valid `v2` seed-1 checkpoints next, beginning with `h8` versus the
incumbent `h16`; advance the winner through `h4` and `sf32`, then confirm the
surviving comparison with seed 2.

The direct refined seed-1 comparison scored `h8 +1.4 +/-14.4` Elo versus h16
(`-0.07/2.20` LLR, `57.5%` LOS, `35.1%` draws) in 1,500 games. The two are
indistinguishable, so stop the bucket-only race and redirect architecture work
to genuinely new tactical information.

### Slider x-ray feature probe

Test one isolated tactical addition on the mature native h16 foundation:
`enyo-sliderxray-v1-rc1` adds only bishop, rook, and queen interactions with
the first occupied piece revealed behind a blocker. It excludes ordinary
direct threats and uses the existing 60,720-row FullThreats index space. Warm
start from `enyo-1.30.0-rc3.nn`, preserving every existing weight and
zero-initializing the new rows. First run only the historically positive
256-superbatch continuation regimen, then test directly against
`enyo-1.30.0-rc3.nn`. RC1 trained the whole network and failed the residual
gate catastrophically, so RC2 is the clean isolation test: freeze all mature
piece rows, the input bias, and every dense weight; train only the zero-init
x-ray rows for 256 superbatches, with no activation penalty or weight decay.
RC2 at `lr=0.00001` preserved every mature tensor but produced no x-ray value
large enough to survive integer export. RC3 therefore changes only the learning
rate to `0.0001`, the smallest already-tested rate expected to cross export
resolution. Stop after RC3 if parity, runtime cost, gates, or Elo is poor; do
not spend a long training dose on a rejected feature.

Use canonical `enyo-{architecture_number}.{promotion_number}.0-rc{iteration}`
run names. Assign one architecture number to each configuration and use `rc1`
and `rc2` for its two random-initialization replicates; record the
human-readable ID and full fields in `hypothesis`, never in the run name.

### Replication and randomness contract

Train two replicates of every configuration. For replicate 1 or 2, every
configuration uses the same declared `init_seed`: `5090001` or `5090002`,
respectively. Seeded initialization derives
an independent ChaCha stream from the seed and tensor name, so differently sized
architectures cannot shift the random stream of another tensor. Record the
effective seed in each run's resolved configuration and provenance.

Data ordering is controlled by the immutable shared
`data/bullet/enyo-architecture-race-pylon-2.8b.bullet` artifact and Bullet's
direct sequential loader, not by a random seed. Build that artifact once, record
its hash, and reuse it without conversion for all 24 runs. The game runner's
`config.json` seed is not a training seed and must not be presented as one.

### Documented execution steps

1. Freeze the field and protocol.
   - Record the twelve IDs, exact architecture JSON, two initialization seeds, corpus
     hash, trainer hash, Enyo engine hash, opening-suite hash, time control, and
     game count before training.
   - Do not add, remove, or modify candidates after observing results. Any later
     idea belongs to a new competition.

2. Verify architecture support.
   - For every configuration, verify trainer construction, checkpoint sizing,
     export, loader metadata, scalar evaluation, SIMD evaluation, and incremental
     accumulator refresh.
   - Require trainer/runtime feature-index parity and scalar/SIMD score parity on
     the fixed suite.
   - Measure start-position and multi-FEN evaluation speed for each architecture.
     A parity or runtime failure makes the configuration ineligible; it is not a
     training loss.

3. Freeze the training input.
   - Verify and record the SHA-256 and size of the pylon source plus the resolved
     offset, limit, and filters.
   - Convert it once to the declared shared Bullet artifact, then record that
     artifact's SHA-256 and row count.
   - Confirm that every run reads identical rows in identical order and processes
     the same number of positions at each checkpoint.
   - Do not regenerate, filter, resample, or relabel the corpus between candidates.

4. Verify the resolved training recipe.
   - Materialize each run from the minimal `build.json` above and its candidate
     `architecture.json`.
   - Reject the launch if any resolved training field differs except the declared
     architecture/feature fields and replicate seed.
   - Confirm there is no origin net and archive the resolved config and provenance.

5. Run all full scratch trainings.
   - Launch exactly one Forge-owned iteration at a time through the repository
     workflow; never duplicate or interfere with active jobs.
   - Train every configuration and both seeds for exactly `196608` superbatches;
     do not eliminate a configuration based on its other replicate's result.
   - Preserve the final checkpoint of every run, including optimizer state,
     hashes, elapsed time, and processed-position count.
   - A failed job is retried only from its verified optimizer checkpoint and
     schedule position. Never restart it under the same run identity with new
     random state.

6. Export and gate every checkpoint.
   - Verify that the quantized export differs materially from initialization and
     that no two candidates are accidentally byte-identical.
   - Run load/search, static, move, feature parity, scalar/SIMD parity, activation,
     clamp, and balanced endgame/high-evaluation audits.
   - Gates detect broken artifacts; static or move metrics do not select or
     promote the winner. Game results decide.

7. Measure fixed-budget results.
   - Test the single final checkpoint from every replicate with the same fixed
     paired-opening sample against one frozen Enyo anchor.
   - Use fixed-size matches, not SPRT early stopping, for ranking.
   - Report seed uncertainty. If a richer HalfKA candidate remains highly
     seed-sensitive, report that the fixed dose did not settle its ranking.

8. Run the final game competition.
   - Test every final checkpoint against the same frozen Enyo anchor using the
     same paired openings with colors reversed.
   - Use a fixed real-time control, identical engine binary/settings, and equal
     hardware. Do not use fixed nodes: slower Reckless or FullThreats evaluation
     must pay its search cost.
   - Run at least 1500 games per replicate/configuration; prefer 3000 when the
     available Forge budget permits. Never stop a weak candidate early.
   - If budget permits a full all-pairs round robin, declare and schedule it
     before results are visible. Otherwise the common-anchor tournament is the
     sole ranking dataset.

9. Analyse all replicates jointly.
   - Fit configuration, seed, and paired-opening effects and report Elo,
     uncertainty interval, evaluation speed, and the probability that each
     configuration is strongest.
   - Do not choose the largest single-run point estimate. A candidate must be
     consistent across seeds and strong under real-time search.
   - Publish all losses, draws, wins, draw rate, Elo, confidence interval, LOS,
     hashes, and exact game counts; do not report only the winning rows.

10. Declare the outcome once.
    - The winner is the configuration with at least 90% estimated probability of
      being strongest whose lower uncertainty bound is no more than 3 Elo below
      every alternative.
    - If no configuration satisfies that rule, declare the statistically tied
      winner set rather than inventing a unique winner.
    - Promotion requires a clean runtime/parity record, a settled learning curve,
      and no material balanced endgame/high-evaluation regression.
    - Record the winner or tied set, complete protocol, and limitations in this
      document. Any subsequent architecture/feature search is a new explicitly
      authorized experiment, not an extension chosen after seeing these results.

## Stage 1: architecture support

Support and verify the practical Enyo-native matrix before starting comparison
training:

- Hidden widths: 512, 768, and 1024.
- Enyo-native input bucket layouts: 10, 16, and 32.
- Feature channels: 11 and 12.
- Output buckets: 1, 4, and 8.
- Explicit architecture metadata in exported nets and strict runtime validation.
- Trainer, exporter, scalar runtime, SIMD runtime, and parity-test support.
- True checkpoint resume preserving weights, Adam momentum and velocity, current
  superbatch, and learning-rate schedule position.

Exit gate: every matrix architecture exports, loads, and produces matching
trainer/runtime evaluations on the parity suite. A stopped short run must resume
to final weights within the measured variation between independent uninterrupted
CUDA runs.

## Stage 2: short training

Train all candidates before comparing them:

1. `enyo-16x12-1024-o8` - control.
2. `enyo-16x12-1024-o4`.
3. `enyo-16x12-1024-o1`.
4. `enyo-16x12-768-o8`.
5. `enyo-16x11-768-o8`.
6. `enyo-10x11-768-o8` - Enyo-derived king map, not a copied engine layout.
7. `enyo-32x11-1024-o8`.
8. `enyo-16x12-512-o1`.

Hold these variables fixed for every candidate:

- Random initialization; never use weights from another engine.
- The same 2.8B-position pylon Bullet file and record order.
- The same batch settings, WDL, filters, initial LR, and final LR.
- The same 65,536-superbatch schedule, stopped at 16,384 for screening.
- A complete optimizer checkpoint saved at superbatch 16,384.

Exit gate: all eight short nets and resumable checkpoints exist, pass static
validation, and have recorded training times and hashes.

## Stage 3: architecture screening

Results:

- `enyo-16x12-1024-o4` beat the control by `+19.1 +/-20.8` Elo over
  1,000 games, with `96.5%` LOS and `32.1%` draws. Advance it to the
  finalist pool.
- `enyo-16x12-1024-o1` lost to the control by `-11.1 +/-19.8` Elo over
  1,000 games, with `13.5%` LOS and `36.8%` draws. Eliminate it.
- `enyo-16x12-768-o8` beat the control by `+31.7 +/-19.7` Elo over
  1,000 games, with `99.9%` LOS and `35.9%` draws. Advance it to the
  finalist pool.
- `enyo-16x11-768-o8` beat the control by `+8.7 +/-20.6` Elo over
  1,000 games, with `79.5%` LOS and `32.1%` draws. Keep it in the
  provisional finalist pool pending the remaining screens.
- `enyo-10x11-768-o8` beat the control by `+28.9 +/-20.2` Elo over
  1,000 games, with `99.8%` LOS and `33.9%` draws. Advance it to the
  finalist pool.
- `enyo-32x11-1024-o8` beat the control by `+29.9 +/-20.5` Elo over
  1,000 games, with `99.8%` LOS and `32.6%` draws. Advance it to the
  finalist pool.
- `enyo-16x12-512-o1` lost to the control by `-7.3 +/-20.4` Elo over
  1,000 games, with `24.1%` LOS and `32.5%` draws. Eliminate it.
- Finalist round robin: `enyo-16x12-768-o8` beat
  `enyo-32x11-1024-o8` by `+13.9 +/-19.8` Elo over 1,000 games, with
  `91.6%` LOS and `35.2%` draws.
- Finalist round robin: `enyo-16x12-768-o8` lost to
  `enyo-10x11-768-o8` by `-5.9 +/-21.1` Elo over 1,000 games, with
  `29.2%` LOS and `35.1%` draws.
- Finalist round robin: `enyo-32x11-1024-o8` lost to
  `enyo-10x11-768-o8` by `-20.9 +/-19.7` Elo over 1,000 games, with
  `1.9%` LOS and `35.2%` draws. Select `enyo-10x11-768-o8` as the
  Stage 3 winner.

- Run 1,000 fixed-protocol games for each candidate against the short-trained
  `enyo-16x12-1024-o8` control.
- Eliminate a candidate after one controlled negative result. Do not immediately
  retry a rejected architecture.
- Advance the strongest three positive or statistically tied candidates.
- Run a three-match, 1,000-game round robin between those finalists.
- Do not run an eight-way all-pairs tournament; the maximum is ten matches.

Exit gate: select one winner. If two finalists remain indistinguishable, advance
both to Stage 4. If no candidate beats the control, the control wins.

## Stage 4: full winner training

- Resume the winner from superbatch 16,384 to 65,536.
- Preserve optimizer state and LR schedule position; do not restart training.
- Resume both finalists only when Stage 3 leaves a genuine tie.

Exit gate: the full net passes export, static, move, and runtime parity checks.

## Stage 5: full validation

Run sequential fixed-size tests:

1. Full winner versus the current fully trained Enyo baseline: 1,000 games.
2. If successful, full winner versus `default.net`: 1,000 games.
3. Only after beating `default.net`, full winner versus
   `~/code/cpp/chess/enyo/net/berserk-9b84c340af7e.nn`: 1,000 games.

The Berserk net is an opponent only. Its weights must never initialize, alter, or
otherwise influence an Enyo net.

Exit gate: establish the validated winner as the new native lineage root, or
record why the existing full Enyo baseline remains champion.

Result: `enyo-10x11-768-o8` scored `-4.9 +/-19.9` Elo with `31.6%`
LOS over 1,000 games against `enyo-scratch-broad-1.0.0-rc1`. The existing
full baseline remains champion; default-net and Berserk tests were not run.

## Stage 6: incremental improvement

- Use `continue_from` for every same-architecture continuation.
- Change one meaningful training variable per rejected experiment.
- After one controlled rejection, move to the next documented hypothesis.
- Preserve a successful regimen and advance only the data slice.
- Run the fixed default-net benchmark periodically to measure absolute progress.

Gap-closing sequence:

1. Train the current architecture from scratch for 196,608 superbatches at
   WDL 0.05 on the existing 2.8B-position pylon Bullet corpus.
2. If rejected, repeat that scratch protocol while changing only WDL to 0.30.
3. If rejected, repeat the winning objective while changing only to the best
   alternate Stockfish corpus.

Each candidate is gated for 1,500 games against the current champion. A
promoted candidate receives the fixed default-net benchmark before further
fine-tuning. Revisit architecture only if these recipe tests fail.

Expected time for the long scratch candidate is about 2.5-3 hours of training
plus its gates and 1,500-game SPRT.

## Experiment ledger

- `enyo-scratch-calibration-1.0.0-rc1` proved that random-init training can
  recover substantial strength and beat the previous fine-tuned lineage.
- `enyo-scratch-broad-1.0.0-rc1` extended the control recipe to the 2.8B pylon
  corpus and is the full 16x12x1024-o8 comparison baseline.
- `enyo-scratch-broad-1.1.0-rc2` promoted float-head training at
  `+1.2 +/-15.2` Elo over 1,500 games.
- `enyo-scratch-broad-1.2.0-rc1` was rejected at `-16.2 +/-15.6` Elo after
  applying the same 1,024-superbatch float-head regimen to the second disjoint
  134,217,728-position slice.
- `enyo-scratch-broad-1.2.0-rc2` reduced the same slice to 256 superbatches
  and was rejected at `-1.4 +/-15.0` Elo over 1,500 games.
- `enyo-scratch-broad-1.2.0-rc3` reduced the second slice to 128 superbatches
  and was promoted at `+8.1 +/-15.2` Elo over 1,500 games.
- `enyo-scratch-broad-1.3.0-rc1` applied that regimen to the third slice and
  was rejected at `-20.4 +/-15.3` Elo over 1,500 games.
- `enyo-scratch-broad-1.3.0-rc2` reduced the third slice to 64 superbatches
  and was rejected at `-18.5 +/-15.3` Elo over 1,500 games.
- `enyo-scratch-broad-1.3.0-rc3` applied 128-superbatch float-head training to
  the fourth slice and was promoted at `+2.6 +/-14.9` Elo.
- Its 500-game default-net checkpoint scored `-215.4 +/-58.0` Elo.
- `enyo-scratch-broad-1.4.0-rc1` applied the same regimen to the fifth slice
  and was rejected at `-3.9 +/-14.7` Elo.
- `enyo-scratch-broad-1.4.0-rc2` switched the fifth slice to input-only
  training and was promoted at `+2.3 +/-14.9` Elo.
- `enyo-scratch-broad-1.5.0-rc1` continued input-only training and was
  promoted at `+5.3 +/-15.1` Elo.
- `enyo-scratch-broad-1.6.0-rc1` was rejected at `+0.5 +/-14.8` Elo because
  its LLR was slightly negative.
- `enyo-scratch-32bucket-1.0.0-rc1` changed only 16 to 32 input buckets and was
  rejected at -6.3 +/-15.0 Elo over 1,500 games. Do not promote or retrain that
  exact architecture in Stage 2.

Game results decide promotion. Static and move gates remain rejection filters,
not promotion evidence.

## 2026-07-14 Quantized Input Export Finding

- The previous `input`-only runs at `lr=5e-6` changed `raw.bin` but did not change `quantised.bin`; exported `.nn` files were therefore byte-identical after the initial v2/container re-export of the 0.48-scaled parent.
- Measured raw deltas stayed below the int16 rounding threshold: 32 superbatches max `0.018`, 128 superbatches max `0.065`, with `0` values >= `0.5`.
- New quantization-aware input sweep launched from `/home/petter/assets/nets/enyo-1.30.0-rc3.nn` on the later broad slice:
  - `enyo-1.31.0-rc10` (`lr=5e-5`): distinct export, gates pass, startpos `+31`, static `mae=304.025`, `corr=0.772137`, `slope=0.207072`.
  - `enyo-1.31.0-rc12` (`lr=1e-4`): distinct export, gates pass, startpos `+111`, static `mae=296.737`, `corr=0.749503`, `slope=0.242578`.
  - `enyo-1.31.0-rc13` (`lr=2e-4`): distinct export, gates pass, startpos `+120`, static `mae=296.229`, `corr=0.748438`, `slope=0.257537`.
- Reference SPRTs versus `enyo-1.28.0-rc16` are queued/running for all three. Treat older input-only Elo as v2 re-export behavior unless the candidate is one of the quantkick exports above.
- `enyo-1.31.0-rc12` was manually stopped versus reference at 246/1500 games after trending clearly bad: `elo=-29.7`, `llr=-0.39/2.20`, LOS `6.1%`. Reject this LR point.

## 2026-07-14 Quantkick Output Follow-up

- Trained output-only calibration on top of real quantized input updates:
  - `enyo-1.31.0-rc14`: gates pass, startpos `+124`, static `mae=292.939`, `corr=0.754848`, `slope=0.302332`.
  - `enyo-1.31.0-rc15`: gates pass, startpos `+42`, static `mae=279.034`, `corr=0.770662`, `slope=0.336410`.
- Reference SPRTs versus `enyo-1.28.0-rc16` are queued for both output follow-ups. Prioritize any candidate that clears champion by SPRT for Berserk testing.
- Reject `enyo-1.31.0-rc13`: manually stopped versus reference at 168/1500 games, `elo=-24.9`, `llr=-0.23/2.20`, LOS `14.7%`.
- Reject `enyo-1.31.0-rc10`: manually stopped versus reference after an early bad trend, `elo=-64.4`, `llr=-0.21/2.20`, LOS `5.7%` at 60 games.
- Added midpoint `enyo-1.31.0-rc11` (`lr=7.5e-5`): gates pass, startpos `+55`, static `mae=297.809`, `corr=0.752078`, `slope=0.231790`.
- Added output follow-up `enyo-1.31.0-rc16`: gates pass, startpos `+60`, static `mae=286.752`, `corr=0.764288`, `slope=0.318094`; reference SPRT queued as `sprt-enyo-1.31.0-rc16.nn-vs-enyo-1.28.0-rc16.nn-1500-20260714-071511`.
- Reject `enyo-1.31.0-rc14`: manually stopped versus reference at 122/1500 games, `elo=-57.5`, `llr=-0.37/2.20`, LOS `8.7%`.
- Reject `enyo-1.31.0-rc15`: manually stopped versus reference at 94/1500 games, `elo=-84.7`, `llr=-0.42/2.20`, LOS `2.9%`.
- Reject `enyo-1.31.0-rc16`: manually stopped versus reference at 140/1500 games, `elo=-34.9`, `llr=-0.28/2.20`, LOS `14.5%`.
- Added `enyo-1.31.0-rc17`: short all-layer fine-tune from 0.48 (`64` superbatches, `lr=1e-5`, trainable `all`), gates pass with startpos `+52`, static `mae=273.550`, `corr=0.771382`, `slope=0.343816`; reference SPRT queued.
- `enyo-1.31.0-rc17` is active versus reference and positive at 360/1500 games: `elo=+19.3`, `llr=0.30/2.20`, LOS `88.9%`.
- `enyo-1.31.0-rc18` gates passed, but its zero-game SPRT was deleted as redundant follow-up queue clutter. Do not queue another same-family all-layer follow-up unless byte/delta screening shows a materially different candidate and the active all-layer SPRT justifies it.
- `enyo-1.31.0-rc19` gates passed, but its zero-game SPRT was deleted as redundant follow-up queue clutter. Do not queue another same-family all-layer follow-up unless byte/delta screening shows a materially different candidate and the active all-layer SPRT justifies it.
- Queue discipline: before any new SPRT, check exported SHA and delta profile against existing candidates; do not queue multiple same-family variants on gates alone.
- `enyo-1.31.0-rc17` completed versus reference: `+3.7 +/-14.3` Elo, `llr=0.13/2.20`, LOS `69.5%`, draw `36.3%`, 1500/1500 games. Do not promote; inconclusive/slightly positive only.

## 2026-07-14 Canonical Rename Cleanup

- Descriptive run/net names from the scale and quantkick work were renamed to canonical `enyo-A.P.0-rcN` names.
- Accepted scale parent is now `enyo-1.30.0-rc3` (`hypothesis`: 0.48 output scale), and `/home/petter/assets/nets/candidate.net` points to it.
- Old descriptive asset and run names were left as symlinks to preserve logs/provenance.
- Next controlled candidate is `enyo-1.31.0-rc20`, using `continue_from= enyo-1.30.0-rc3`, `reference= enyo-1.30.0-rc3`, `trainable=all`, `superbatches=96`, `lr=1e-5`, `wdl=0.05`.

## 2026-07-14 Diagnosis-First Recovery Plan

The current 16-bucket, 12-channel, 1024-wide Enyo-native architecture is
frozen. Do not start another architecture experiment: the relevant 768-wide
screen was already tested, FullThreats was already tested, and the recent
output-scaling, quantization-aware, trainable-scope, dose, and data-slice
lanes did not close the Berserk gap.

The rejected `enyo-1.31.0-rc20` through `rc34` sequence is not evidence for
another fine-tuning sweep. It varied training scope, dose, and later data
slices around the same accepted parent; `rc34` also failed the Stockfish gate
at `-173.6` Elo and the parent-relative test was negative before the redundant
smoke rerun. No new net is permitted until the cause is isolated.

Next work, in order:

1. Reproduce Enyo, Berserk, and Stockfish on the identical validation FENs and
   compare raw network output, engine scaling, clamping, side-to-move, phase,
   and output-bucket distributions.
2. Audit the standard feature/data path end to end: Bullet features versus
   runtime features, labels and score scaling, WDL transformation, ply and
   material filters, and training/validation distribution differences.
3. Verify that a controlled training change survives quantised export and is
   materially present in the runtime net. A gate pass without a quantised
   delta is not a valid experiment.
4. Only after a concrete mismatch or deficiency is identified, implement one
   correction with architecture, reference, and unrelated training variables
   held fixed.
5. Require improved relevant static buckets and one smoke result before a
   single 1500-game SPRT. Do not queue variants or automatically continue a
   rejected lane.

The first measured static clue remains output distribution, not architecture:
Enyo's fitted scale was about `0.52` against Berserk, final-score clamps were
`14/500` versus `1/500`, and endgame/output buckets carried the largest error.
Global scaling recovered only about 30 Elo, so calibration is a symptom and
not a sufficient fix. The audit must determine whether the remaining error
comes from labels, sampling, optimisation targets, quantisation, or runtime
evaluation.
## 2026-07-14 Diagnosis Audit Results

- The accepted scaled net `enyo-1.30.0-rc3` was compared with Berserk and
  Stockfish on 500 identical FENs from the validation corpus using the same
  Enyo engine.
- Aggregate static agreement is already strong: Berserk correlation `0.9105`,
  fitted slope `0.9487`, MAE `90.26`; Stockfish correlation `0.9109`, fitted
  slope `0.9138`, MAE `82.17`.
- Remaining static error is concentrated in endgames and high evaluations:
  Berserk endgame MAE `188.87` and high-evaluation MAE `202.71`; Enyo has
  `3/500` clamped outputs versus Berserk `1/500`.
- This rules out a whole-network scale or obvious standard feature-index failure
  as the explanation for the game-strength gap. Global output scaling is not
  an adequate next training hypothesis.
- `cargo test --manifest-path tools/bullet/spike_trainer/Cargo.toml --features
  cuda --bin train` passes all 20 tests, including runtime feature-index parity,
  data slicing, provenance, resume, hidden-width conversion, and export sizing.
- The standard feature/export contract is therefore not yet implicated. The
  next audit target is the training target and sample distribution: compare the
  actual Bullet source metadata and effective target transforms against the
  validation population, with special attention to endgame/high-eval rows.
- No candidate training run is authorized from these results. A concrete target
  or sampling defect must be identified first.

Concrete current candidate:

- Aborted `enyo-1.31.0-rc35` before game testing because its new slice made
  the data comparison non-isolated.
- `enyo-1.31.0-rc36` keeps `architecture.json`, parent, WDL, optimizer, dose,
  and trainable scope fixed, and changes only the data regime back to the
  accepted parent's `training_data_pylon.binpack` with `min_ply=24`.
- It uses the accepted parent's documented pylon range starting at offset
  `800,000,000` (100M positions for this controlled run).
- This is the first post-audit candidate and is eligible for one smoke test and
  one 1500-game SPRT only after training/export and static gates pass.

## 2026-07-15 50k target/runtime audit

- Built a deterministic 50,000-position corpus balanced across opening, middlegame, late, and endgame (12,500 each), and compared champion enyo-1.31.0-rc42, Berserk, and Stockfish with phase, absolute-eval, and output-bucket grouping.
- Largest measured mismatch is endgame/high-eval output saturation: Stockfish-vs-champion endgame MAE 285.11 cp with 2,972/12,500 runtime clamps (23.8%); the 800+ group has MAE 432.99 cp with 3,269/8,397 clamps (38.9%). Input clipping is 0%, so this is an output target/runtime-limit problem, not input saturation.
- Verified the target/runtime contract: side orientation, centipawn units, WDL blend, phase normalization, output-bucket selection, and export quantization. Added deterministic contract tests and retained trainer feature-index parity tests.
- Corrected the largest mismatch by clamping training targets to the runtime +/-2045 cp limit before inverse phase normalization. The residual gate now runs the balanced 50k audit and requires endgame, 300-799 cp, and 800+ cp MAE and slope-distance improvement before any SPRT.
- The unchanged-candidate gate test rejects with exit 1. No SPRT has been launched for rc7; its build config is staged as the single controlled follow-up.
