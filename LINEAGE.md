# Enyo 5 Lineage
  ┌───────────────┬───────────┬────────────────┐
  │    Engine     │ Mean eval │ Spread (stdev) │
  ├───────────────┼───────────┼────────────────┤
  │ Enyo rc36     │ 115 cp    │ 123 cp         │
  ├───────────────┼───────────┼────────────────┤
  │ Stockfish net │ 70 cp     │ 59 cp          │
  ├───────────────┼───────────┼────────────────┤
  │ Berserk net   │ 45 cp     │ 48 cp          │
  └───────────────┴───────────┴────────────────┘


`Parent SPRT` is against the immediately preceding selected net.
`SF` is the fixed 4,000-game benchmark against `nn-1a298aa575a0.nnue`

Architecture 5: SCReLU dense activation (`relu-screlu-residual`).
Architecture 6: Architecture 5 plus the activated L2→output skip.
Architecture 7: Architecture 6 plus 32 input buckets.
Architecture 16: Architecture 7 plus FullThreats (initialize_from warm start via
the fixed `add_full_threat_rows`; smoke diagnostic confirmed 89.3% nonzero
threat-row coverage post-quantization and passed distinct_net/startpos/static_eval).
Architecture 8: Architecture 7 plus Stockfish-exact FullThreats (v12 export
contract and matching 59,808-row feature map).
Current selected parent: `enyo-7.5.0-rc36` (rc74 tie-break promotion reverted
2026-09-17 after a wide direct SPRT against its parent; see below).
Flagged 2026-09-09: rc37/rc38 continued the mechanical "next full dose,
fresh slice" pattern for six more iterations after the fixed-slice
superbatches sweep (rc30-rc32) had already made the point that blind
data-offset incrementing without a reason isn't a real hypothesis — this
was called out directly and should not recur.
rc27 was independently verified genuinely better than `enyo-7.5.0-rc17` via
a wider 8000-game direct SPRT (LLR 2.31/2.20, elo+5.4, los=97.5%) after the
default 4,000-game samples on this whole chain proved too noisy (~7 Elo CI)
to resolve an effect this small — individual SF points should not be
over-interpreted without a comparably-sized paired check.
Reverted 2026-09-10: `enyo-7.5.0-rc38` was promoted on SF tie-break only
(-117.9 ±6.7 vs -122.7 ±6.5 for rc36) with its own direct parent SPRT
already unfavorable (-4.9 ±6.2, LLR -1.58/2.20, at 4,000/4,000 games,
inconclusive). Per the standing rule that this weakest evidence tier gets
an independent wide-SPRT check (as `rc27` got before it), a 12,000-game
ceiling direct SPRT of `rc38` vs `rc36` was run: it resolved cleanly against
`rc38` — elo=-3.8 ±3.8, LLR -2.29/2.20 (-104%, H0), los=2.6%, draw=55.9%,
games=10,000/12,000. `rc38`'s SF-tie-break promotion does not hold up under
verification; selected parent reverts to `rc36`. `rc38`'s evidence and net
are retained for the record but it is not a valid parent.
Independent direct check against the active `rc27` parent completed
2026-09-17: `rc38` was rejected at H0, elo=-6.8 ±7.3, LLR -2.26/2.20,
los=3.3%, draw=52.2%, games=2,962/10,000 (aggregate SPRT stopped at the
boundary; 5,000/5,000 shards completed, zero failures). This confirms that
the higher fixed-Stockfish score for `rc38` does not justify promotion over
`rc27`.
Reserved: `enyo-7.5.0-rc39` on pwa-llm (single-variable continuation from
`enyo-7.5.0-rc36`: final LR 0.000005 → 0.00001, everything else held at
baseline; historically-grounded lever — Enyo-5 validated wins in both
directions on final LR, `0.000005 → 0.000010` at row 3 and `0.000010 →
0.000005` at row 14 — unlike initial LR and dose, both already ruled out
this chain).
Rejected: `enyo-7.5.0-rc39` (final LR 0.000005 → 0.00001; -9.9 ±7.4, LLR
-2.32/2.20, H0, los=0.4%, games=2836/4000, commit `81131e2e`).
Reserved: `enyo-7.5.0-rc40` on pwa-llm (single-variable continuation from
`enyo-7.5.0-rc36`: initial LR 0.004 → 0.003, everything else held at
baseline, same data slice — no advance, this follows a rejection not an
acceptance. Every LR probe on this chain so far has been an *increase*
(initial LR 0.004→0.005: -25.1 Elo; final LR 0.000005→0.00001: -9.9 Elo,
just above); a decrease is the only untested LR direction left. Weaker
grounding than the final-LR pick — no validated win in this direction on
record, just elimination of the alternatives — but the pattern of two
consecutive failed increases is consistent with 0.004 already being past
optimal rather than under it).
Rejected: `enyo-7.5.0-rc40` (initial LR 0.004 → 0.003; -2.5 ±6.0, LLR
-0.87/2.20 (inconclusive), los=20.7%, draw=55.6%, games=4000/4000, Stockfish
tie-break also unfavorable (-126.9 vs -122.7), commit `e274374a`).
Reserved: `enyo-7.5.0-rc41` on pwa-llm (single-variable continuation from
`enyo-7.5.0-rc36`: `activation_l1` 0.0 → 0.0001, everything else held at
baseline, same data slice — rejection-driven, no advance. First iteration
in this chain touching a genuinely untested dimension rather than another
LR/WDL/dose nudge (all of which are now closed off in both directions).
`activation_l1` is a standard NNUE-specific technique — L1 penalty on the
clipped feature-transformer activations (`spike_trainer/src/main.rs:1239`,
`activation_l1 / (2*hidden)` normalized), used in the wider NNUE-training
community specifically to reduce int8/int16 quantization error, not a
generic regularizer. Never touched anywhere in this repo's history (always
0.0 in every prior build.json). Magnitude (1e-4) is a literature-typical
starting point, not calibrated to this trainer's exact scale — if it's
badly off, the static_eval/residual gates should catch a degraded net
before the SPRT stage runs at all).
Rejected: `enyo-7.5.0-rc41` (activation_l1 0.0 → 0.0001; -4.6 ±6.2, LLR
-1.35/2.20 (inconclusive), los=7.2%, draw=55.4%, games=4000/4000, Stockfish
tie-break also unfavorable (-129.1 vs -122.7), commit `5fa4e2de`).
Reserved: `enyo-7.5.0-rc42` on pwa-llm (single-variable continuation from
`enyo-7.5.0-rc36`: `weight_decay` 0.0 → 0.01, everything else held at
baseline, same data slice. Never touched anywhere in this lineage (always
0.0). Unlike the `activation_l1` guess, the magnitude has concrete
grounding: standard decoupled AdamW-style decay (`w -= lr*decay*w` per
step, `tools/bullet/bullet-patched/crates/trainer/src/optimiser/decay.rs`),
and 0.01 is literally the bullet crate's own built-in default
(`optimiser/adam.rs:29`) — this project's `defaults.json` overrides it to
0.0 with no record of that override ever being tested. Different mechanism
from `activation_l1` (global weight-norm shrinkage vs FT-activation
sparsity), so its rejection doesn't pre-empt this one).
Rejected: `enyo-7.5.0-rc42` (weight_decay 0.0 → 0.01; catastrophic, -87.2
±22.5, LLR -2.46/2.20 (-112%, H0), los=0.0%, games=350/4000, early smoke
triage, commit `e3597280`). In hindsight the magnitude was badly
miscalibrated for this use: 768 superbatches × 64 batches = 49,152 steps of
`w *= (1 - lr*decay)` ≈ `(1-0.00004)^49152` ≈ 0.14 — roughly 86% cumulative
weight shrinkage over the run. The bullet crate's 0.01 default is presumably
sane for training from scratch over a different step count/schedule, not
for a short (768-superbatch) continuation fine-tune off an already-converged
checkpoint. A much smaller value (if this is revisited, think 1e-4 to 1e-5,
not 0.01) would be needed to avoid dominating the loss at this scale.

Reserved: `enyo-7.5.0-rc43` on pwa-llm (final conventional fallback before
the compact-topology architecture work; single-variable continuation from
`enyo-7.5.0-rc36`: `weight_decay` 0.0 → 0.0001, everything else held at
baseline, same data slice. This tests the small decay scale suggested by
rc42's postmortem: about 2% cumulative shrinkage over 49,152 steps rather
than rc42's roughly 86%).
Rejected: `enyo-7.5.0-rc43` (`weight_decay` 0.0 → 0.0001; parent SPRT
inconclusive but unfavorable: −6.1 ±6.1 Elo, LLR −1.99/2.20, los=2.6%,
draw=55.0%, games=4000/4000; fixed-SF tie-break also unfavorable:
−131.0 ±7.0 versus rc36 −122.7 ±6.5, games=4000/4000, commit `145d150e`).
Reserved: `enyo-7.5.0-rc44` on pwa-llm (single-variable continuation from
`enyo-7.5.0-rc36`: `weight_decay` 0.0 → 0.00001, everything else held at
baseline and the same data slice. This is the remaining scale explicitly
identified by rc42's postmortem: approximately 0.2% cumulative AdamW shrinkage
over 49,152 steps, rather than rc43's approximately 2% or rc42's 86%).
Rejected: `enyo-7.5.0-rc44` (`weight_decay` 0.0 → 0.00001; parent SPRT
inconclusive but unfavorable: −4.3 ±6.3 Elo, LLR −1.45/2.20, los=9.4%,
draw=54.7%, games=4000/4000; fixed-SF tie-break also unfavorable:
−126.4 ±6.7 versus rc36 −122.7 ±6.5, games=4000/4000).
Reserved: `enyo-7.5.0-rc45` on pwa-llm (single-variable continuation from
`enyo-7.5.0-rc36`: `trainable` `all` → `input`, everything else held at
baseline and the same data slice. The stronger same-architecture historical
benchmark `enyo-1.32.0-rc10` used `trainable: input`; this directly tests that
training-scope difference without adopting its foreign weights or changing
architecture).
Rejected: `enyo-7.5.0-rc45` (`trainable` `all` → `input`; parent SPRT
inconclusive: +1.0 ±6.1 Elo, LLR 0.35/2.20, los=63.0%, draw=53.3%,
games=4000/4000; fixed-SF tie-break unfavorable: −130.0 ±6.8 versus rc36
−122.7 ±6.5, games=4000/4000).
Reserved: `enyo-7.5.0-rc46` on pwa-llm (single-variable continuation from
`enyo-7.5.0-rc36`: `superbatches` 768 → 128, everything else held at
baseline and the same data slice. The stronger same-architecture historical
benchmark `enyo-1.32.0-rc10` used 128 superbatches; this isolates its
continuation-length difference after rc45 ruled out its input-only scope).
Rejected: `enyo-7.5.0-rc46` (`superbatches` 768 → 128; parent SPRT
unfavorable: −6.4 ±6.3 Elo, LLR −2.19/2.20, los=2.2%, draw=54.5%,
games=3848/4000; fixed-SF tie-break also unfavorable: −125.3 ±7.0 versus
rc36 −122.7 ±6.5, games=4000/4000).
Reserved: `enyo-7.5.0-rc47` on pwa-llm (single-variable continuation from
`enyo-7.5.0-rc36`: `superbatches` 768 → 512, everything else held at
baseline and the same data slice. This is the intermediate historical Pylon
dose after rc46 ruled out the much shorter 128-superbatch continuation).
Rejected: `enyo-7.5.0-rc47` (`superbatches` 768 → 512; parent SPRT
inconclusive: +3.7 ±6.1 Elo, LLR 1.21/2.20, los=88.3%, draw=56.0%,
games=4000/4000; fixed-SF tie-break unfavorable: −127.0 ±6.7 versus rc36
−122.7 ±6.5, games=4000/4000).
Reserved: `enyo-7.5.0-rc48` on pwa-llm (single-variable continuation from
`enyo-7.5.0-rc36`: `superbatches` 768 → 640, everything else held at
baseline and the same data slice. This midpoint interpolates rc47's
parent-SPRT signal at 512 superbatches against the 768-superbatch baseline).
Rejected: `enyo-7.5.0-rc48` (`superbatches` 768 → 640; parent SPRT
inconclusive but unfavorable: −2.3 ±6.1 Elo, LLR −0.72/2.20, los=23.3%,
draw=55.1%, games=4000/4000; fixed-SF tie-break also unfavorable:
−124.7 ±6.8 versus rc36 −122.7 ±6.5, games=4000/4000). The 128/512/640
dose probes show no reproducible fixed-SF improvement; close this branch.
Reserved: `enyo-7.5.0-rc49` on pwa-llm (single-variable continuation from
`enyo-7.5.0-rc36`: `activation_l1` 0.0 → 0.00001, everything else held at
baseline and the same data slice. rc41's 0.0001 activation penalty was
unfavorable; this tenfold smaller scale retains a light quantization-aware
regularization signal rather than retesting that magnitude).
Rejected: `enyo-7.5.0-rc49` (`activation_l1` 0.0 → 0.00001; parent SPRT
inconclusive: +3.2 ±6.3 Elo, LLR 1.01/2.20, los=84.2%, draw=55.5%,
games=4000/4000; fixed-SF tie-break unfavorable: −130.2 ±6.9 versus rc36
−122.7 ±6.5, games=4000/4000).
Reserved: `enyo-7.5.0-rc50` on pwa-llm (single-variable continuation from
`enyo-7.5.0-rc36`: `activation_l1` 0.0 → 0.00005, everything else held at
baseline and the same data slice. This midpoint tests the remaining
regularization scale between baseline and rc41's rejected 0.0001 penalty).
Inconclusive: `enyo-7.5.0-rc50` (`activation_l1` 0.0 → 0.00005; parent
SPRT: −1.1 ±6.2 Elo, LLR −0.42/2.20, los=36.1%, draw=52.8%,
games=4000/4000). Fixed-SF report: −134.2 ±6.8 versus rc36 −122.7 ±6.5,
games=4000/4000. Retain rc36: fixed-SF does not select the parent.
Reserved: `enyo-7.5.0-rc51` on pwa-llm (single-variable continuation from
`enyo-7.5.0-rc36`: `activation_l1` 0.0 → 0.00002, with all other baseline
settings and the same data slice held fixed. This refines rc49's positive but
inconclusive +3.2 ±6.3 parent-relative estimate below rc50's unfavorable
0.00005 scale).
Inconclusive: `enyo-7.5.0-rc51` (`activation_l1` 0.0 → 0.00002; parent
SPRT: −3.1 ±6.3 Elo, LLR −1.08/2.20, los=16.6%, draw=54.4%,
games=4000/4000). Fixed-SF report: −125.9 ±6.8 versus rc36 −122.7 ±6.5,
games=4000/4000. Retain rc36: fixed-SF does not select the parent.

Reserved: `enyo-7.5.0-rc52` on pwa-llm (single-variable continuation from
`enyo-7.5.0-rc36`: `superbatches` 768 → 576; all other baseline settings
and the same data slice are held fixed. This locally follows rc47's +3.7
±6.1 estimate at 512 and rc48's −2.3 ±6.1 result at 640).
Inconclusive: `enyo-7.5.0-rc52` (`superbatches` 768 → 576; parent SPRT:
−4.5 ±6.2 Elo, LLR −1.51/2.20, los=7.8%, draw=54.8%, games=4000/4000).
Fixed-SF report: −127.2 ±7.1 versus rc36 −122.7 ±6.5, games=4000/4000.
Retain rc36: fixed-SF does not select the parent.
Reserved: `enyo-7.5.0-rc53` on pwa-llm (independent single-variable
replication from `enyo-7.5.0-rc36`: `superbatches` 768 → 512; all other
baseline settings and the same data slice held fixed. The loader's
time-seeded shuffling produces an independent training trajectory, directly
testing whether rc47's +3.7 ±6.1 result reproduces).
Rejected: `enyo-7.5.0-rc53` (`superbatches` 768 → 512; parent SPRT:
−7.5 ±6.4 Elo, LLR −2.33/2.20, los=1.1%, draw=54.0%, games=3704/4000,
H0). The independent replication rejects rc47's positive estimate; close
the superbatch branch.
Reserved: `enyo-7.5.0-rc54` on pwa-llm (independent single-variable
replication from `enyo-7.5.0-rc36`: `activation_l1` 0.0 → 0.00001; all
other baseline settings and the same data slice held fixed. The loader's
time-seeded shuffling produces an independent trajectory, directly testing
whether rc49's +3.2 ±6.3 result reproduces).
Inconclusive: `enyo-7.5.0-rc54` (`activation_l1` 0.0 → 0.00001, independent
replication; parent SPRT: −3.5 ±6.0 Elo, LLR −1.05/2.20, los=12.9%,
draw=54.5%, games=4000/4000). The independent replication does not confirm
rc49's positive estimate either — both `activation_l1` and `superbatches`
"promising but inconclusive" branches (rc47/rc53, rc49/rc54) failed to
reproduce under independent trajectories. Treat both as noise, not signal;
close both branches. Retain `rc36`.

Every conventional single-variable lever near `rc36` has now been swept:
initial LR (both directions), final LR (up), WDL (down, historical),
dose/superbatches (128/512/576/640/768), `activation_l1` (0.00001/0.00002/
0.00005/0.0001), `weight_decay` (0.00001/0.0001/0.01), and `trainable`
(`input`). None held up, including both replicated "promising" point
estimates. `rc36` remains the selected parent.

Reserved: `enyo-7.5.0-rc55` on pwa-llm (single-variable continuation from
`enyo-7.5.0-rc36`: `wdl` 0.025 → 0.035, everything else held at baseline
and the same data slice. WDL has only been probed downward in this lineage
(0.025 → 0.015, neutral, pre-dating this session; 0.05 → 0.025 was a
genuine win at architecture 7.4). This completes the local WDL neighborhood
before considering anything beyond conventional hyperparameters — the
`enyo-1.32.0-rc10` benchmark's other difference, "recalibrated nodes5000pv2
labels," is a corpus-relabeling change, not a build.json-tunable variable,
and is out of scope for a single-iteration probe).
Rejected: `enyo-7.5.0-rc55` (`wdl` 0.025 → 0.035; parent SPRT unfavorable:
−7.2 ±6.2 Elo, LLR −2.27/2.20 (H0), los=1.2%, draw=54.2%, games=3820/4000,
commit `40690703`). Closes the WDL neighborhood: both directions (0.015,
0.035) are now worse than 0.025; `rc36`'s WDL is a local optimum.

Every conventional single-variable lever near `rc36` is now exhausted:
initial LR (up and down), final LR (up), WDL (up and down),
dose/superbatches (128/512/576/640/768), `activation_l1` (four magnitudes,
including a failed independent replication of its one promising estimate),
`weight_decay` (three magnitudes), and `trainable` (`input`). None held up.
`rc36` remains the selected parent. Further progress from here likely
requires something outside build.json-tunable hyperparameters — e.g. a
genuinely different corpus/label source, not another magnitude sweep.

Inconclusive: `enyo-7.5.0-rc56` (data offset 2,100,000,000 → 0, everything
else at baseline, off `rc36`; parent SPRT: -1.0 ±6.1 Elo, LLR -0.26/2.20,
los=36.9%, draw=54.1%, games=4000/4000, commit `e3cb25f1`). No signal either
way from the fresh slice at offset 0.
Rejected: `enyo-7.5.0-rc57` (`lr` 0.004 → 0.010, off `rc36`, offset 0;
catastrophic — parent SPRT: -216.7 ±38.6 Elo, LLR -2.81/2.20 (H0),
los=0.0%, draw=46.8%, games=186/4000, early-triaged, commit `67ff7724`).
Consistent with, not contradicting, the earlier finding that even a modest
LR increase (0.004→0.005) was clearly bad (-25.1 Elo) — this was a much
larger untested step size into already-known-bad territory, not a new
result.
Inconclusive: `enyo-7.5.0-rc58` (`wdl` 0.025 → 0.05, `lr` 0.004 → 0.001,
offset 0, off `rc36` — reverting both to pre-tuning historical defaults on
a fresh data slice; three variables changed at once, not a clean
single-variable probe. Parent SPRT inconclusive: +3.8 ±6.0 Elo, LLR
1.23/2.20 (56%), los=89.6%, draw=56.2%, games=4000/4000; Stockfish tie-break
unfavorable: -129.4 ±6.8 vs `rc36` -122.7 ±6.5, commit `fe1dae01`). The
positive point estimate and high los% echo the same "promising but
inconclusive" shape `rc47` and `rc49` had, both of which failed independent
replication — not a result to act on without isolating which of the three
changed variables (if any) is responsible, and a replication check before
trusting the point estimate at all. `rc36` remains the selected parent.
Inconclusive: `enyo-7.5.0-rc59` (`wdl` 0.025 → 0.08, `lr` 0.004 → 0.001,
offset 0, off `rc36` — same three-variable pattern as `rc58`, pushing `wdl`
further. Parent SPRT inconclusive: +4.9 ±6.1 Elo, LLR 1.78/2.20 (81%, closer
to H1 than rc58's 56% but still short), los=94.0%, draw=54.0%,
games=4000/4000; Stockfish tie-break unfavorable: -126.2 ±6.8 vs `rc36`
-122.7 ±6.5, commit `b7a91c53`). Two consecutive positive-but-short-of-H1
point estimates in the same direction (rc58 los=89.6%, rc59 los=94.0%) as
`wdl` rises is suggestive, but still confounded by the same offset+lr
changes both times, and still losing the SF tie-break both times. `rc36`
remains the selected parent.
Inconclusive: `enyo-7.5.0-rc60` (`wdl` 0.025 → 0.20, `lr` 0.004 → 0.001,
offset 0, off `rc36` — same pattern, wdl pushed much further. Parent SPRT
inconclusive: +1.3 ±6.1 Elo, LLR 0.55/2.20 (25%), los=66.3%, draw=54.2%,
games=4000/4000; Stockfish tie-break unfavorable: -124.3 ±6.8 vs `rc36`
-122.7 ±6.5, commit `cd15f835`). Breaks the rc58→rc59 rising-los% pattern
(89.6%→94.0%→66.3%) — consistent with that trend having been noise around
a small/zero effect rather than a real monotonic wdl relationship. Three
consecutive losses on the SF tie-break in this offset-0/lr-0.001/wdl-up
branch. `rc36` remains the selected parent.
Rejected: `enyo-7.5.0-rc61` (`wdl` 0.025 → 0.50, `lr` 0.004 → 0.001,
offset 0, off `rc36` — same branch, wdl pushed to an extreme. Parent SPRT
unfavorable: -11.8 ±8.0 Elo, LLR -2.25/2.20 (H0), los=0.2%, draw=54.2%,
games=2322/4000, commit `29f0bec6`). Closes the branch with a clear
downturn: `wdl` 0.05/0.08/0.20/0.50 → elo +3.8/+4.9/+1.3/-11.8, i.e. a mild
peak somewhere near 0.05-0.08 falling off to clearly bad by 0.5 — consistent
with noise around a small/null effect in the 0.05-0.20 range rather than a
real exploitable signal, not a lever worth pursuing further at these
(offset 0, lr 0.001) settings. `rc36` remains the selected parent.
Inconclusive (unreconciled): `enyo-7.5.0-rc62` (`wdl` 0.025 → 0.20, `lr`
0.004 → 0.005, offset 0, off `rc36`). Parent SPRT confirmed complete via
`forge status enyo-7.5.0-rc62-sprt-4000-20260914-123817`: +1.7 ±6.0 Elo,
LLR 0.61/2.20 (28%), los=71.4%, draw=55.8%, games=4000/4000 — but `rc63`
was launched before the Stockfish tie-break/commit step ran, so no git
commit exists for this result; recorded here from the live Forge run for
completeness. Since it never crossed H1 either way, the missing tie-break
does not change the outcome: not promoted.
Rejected: `enyo-7.5.0-rc63` (`wdl` 0.025 → 0.50, `lr` 0.004 → 0.005,
`superbatches` 768 → 1536, off `rc36`, offset 0 — three variables at once,
each already independently suspect: `wdl=0.5` already rejected alone in
`rc61`, `lr=0.005` already rejected alone much earlier this chain, doubled
dose untested at this exact value. Parent SPRT unfavorable: -20.2 ±10.3
Elo, LLR -2.21/2.20 (H0), los=0.0%, draw=54.3%, games=1324/4000, commit
`91664134`). Not informative about any one variable given the confound;
consistent with the individual components already being bad. `rc36`
remains the selected parent.
Inconclusive: `enyo-7.5.0-rc64` (same as `rc63` but `wdl` 0.50 → 0.20, off
`rc36`, offset 0, `lr=0.005`, `superbatches=1536` — dropping wdl back into
the earlier mild/inconclusive range while keeping doubled dose and the
higher lr. Parent SPRT: -1.6 ±6.0 Elo, LLR -0.53/2.20 (-24%), los=30.6%,
draw=54.9%, games=4000/4000; Stockfish tie-break unfavorable: -129.1 ±6.9 vs
`rc36` -122.7 ±6.5, commit `4bc07408`). Back to no-signal at this combination
too; still confounded (lr+dose+wdl all off baseline at once). `rc36` remains
the selected parent.
Inconclusive: `enyo-7.5.0-rc65` (`continue_from` switched to `enyo-7.5.0-rc27`
instead of `rc36` — a different, earlier point in the lineage; hypothesis
"next full dose off rc27, fresh slice" at baseline hyperparameters,
offset=2,000,000,000. Parent SPRT vs `rc27`: -1.7 ±6.2 Elo, LLR -0.55/2.20
(-25%), los=30.1%, draw=54.7%, games=4000/4000; Stockfish tie-break
unfavorable: -127.3 ±6.9 vs `rc27`'s own -123.5 ±6.9, commit `4f18e5d7`).
Note this compares against `rc27`, not the currently-selected `rc36` — does
not bear on `rc36`'s status either way, and does not itself become a new
parent since it lost its own comparison. `rc36` remains the selected parent.
Void: `enyo-7.5.0-rc66` ("next full dose off rc27, fresh slice", baseline
hyperparameters, offset=1,800,000,000; trained and passed static_eval, but
no Forge SPRT run was ever launched for it — abandoned before any games
were played, no result exists. `enyo-7.5.0-rc67` reused this same offset
next.).
Rejected: `enyo-7.5.0-rc67` ("increase lr 0.001": `lr` 0.004 → 0.005, off
`rc27`, offset=1,800,000,000, same slice `rc66` never tested. Parent SPRT
vs `rc27` unfavorable: -10.2 ±7.3 Elo, LLR -2.23/2.20 (H0), los=0.3%,
draw=55.2%, games=2648/4000, commit `524c3504`). Consistent with every
other `lr` 0.004→0.005 probe this session (`rc57`'s more extreme 0.010 was
also bad; the original rc36-branch 0.004→0.005 test was -25.1 Elo) — this
direction is thoroughly closed regardless of which parent it's tested from.
`rc36` remains the selected parent.
Reverted 2026-09-17: `enyo-7.5.0-rc74` (off `rc27`, fresh slice,
`superbatches` 768 → 1536 and `lr` 0.004 → 0.0015 as the deliberately
combined recipe) initially had parent SPRT +5.0 ±6.3 Elo, LLR 1.56/2.20,
los=93.9%, draw=54.5%, games=4000/4000; its fixed-SF tie-break was -120.6
±6.8 versus `rc27` -123.5 ±6.9. A required wide direct SPRT against `rc27`
then resolved H0: elo=-1.9 ±4.5, LLR -2.24/2.20, los=19.9%, draw=55.0%,
games=7,226/10,000, zero failures. The tie-break promotion does not hold;
the canonical selected parent remains `rc36`.
Rejected: `enyo-7.5.0-rc75` (off `rc27`, fresh slice, `lr` 0.004 → 0.001).
The 10,000-game parent SPRT was inconclusive: +0.3 ±3.9 Elo, LLR 0.42/2.20,
los=56.3%, draw=55.2%, zero failures. The full fixed-SF tie-break was
unfavorable: -131.8 ±6.9 versus `rc27` -123.5 ±6.9, games=4000/4000, zero
failures. It is not promoted.

Void: `enyo-15.0.0-rc1` (legacy-direct-16x12-512; abandoned uncommitted, never SPRT-tested).
Void: `enyo-16.0.0-rc1` (FullThreats via `initialize_from`, warm-start
coverage and gates verified clean beforehand; -97.8 ±17.4, LLR -5.00/2.20, H0).
Reserved: `enyo-16.0.0-rc2` on pwa-5090 (FullThreats scratch root, no
continue_from/initialize_from, to isolate whether rc1's rejection was the
feature or the initialize_from-onto-converged-net methodology).
Void: `enyo-16.0.0-rc3` (two-host local-SGD smoke on pwa-llm + pwa-hak:
2,048 superbatches per host, 16/16 sync rounds including final sync, identical
final quantised SHA-256 `23dd64332115a9c1512b3abdbb3164f15d1fa4086ba87cc43e3408e24bb5f8b3`,
finite weights/loss; 0.86x effective throughput at sync_every=128, no SPRT or
promotion; evidence retained under `~/tmp/enyo-dist-smoke-rc3/` on both hosts).
Void: `enyo-16.0.0-rc4` (two-host local-SGD throughput probe on pwa-llm +
pwa-hak: 512 superbatches per host, one final sync, identical final quantised
SHA-256 `865285d7e419544aa5722ce97afe8cf52f2b8a040d89101ec6b1fdc21c1fb9e3`,
finite weights/loss; 1.44x conservative effective throughput at
`sync_every=512`, no SPRT or promotion; evidence retained under
`~/tmp/enyo-dist-smoke-rc3/` on both hosts).
Reserved: `enyo-8.0.0-rc1` on pwa-llm (Forge coordinator; pwa-llm,
pwa-5090, and pwa-hak training ranks; Stockfish-exact FullThreats v12
warm-started from `enyo-7.4.0-rc1`).
Reserved/reused: `enyo-7.5.0-rc1` on pwa-llm (Forge coordinator; data-only
continuation from `enyo-7.4.0-rc1` using the Forge-labeled LC0 Test91
root-Q expected-score corpus). The first execution was voided at the
start-position integrity gate before games: its V6 conversion incorrectly
used P(win) rather than P(win) + P(draw)/2. The replacement corpus uses
(1 + root_q) / 2 and is repartitioned into 1,600 chunks.
Rejected: `enyo-7.5.0-rc1` (LC0 Test91 v2 raw root-Q targets; -70.4 ±21.0
Elo, LLR -2.20/2.20 at 380/4,000 games, H0). The corpus omitted Enyo runtime
score normalization, so this result is retained as invalid calibration
evidence and is not a parent.
Reserved: `enyo-7.5.0-rc2` on pwa-llm (Forge coordinator; data-only
continuation from `enyo-7.4.0-rc1` using the verified LC0 Test91 v3 corpus,
which applies Enyo runtime score normalization to the identical v2 records).
Void: `enyo-7.5.0-rc3` (25 Aug duplicate-name reuse; −64.7 Elo,
LLR −2.51/2.20. Its evidence and net are retained, but it is not valid
lineage evidence: it also changed dose, trainability, final-LR schedule, and
the corpus relative to `enyo-7.4.0-rc1`).
Rejected: `enyo-7.5.0-rc8` (−46.3 Elo, CI 16.0, LLR −2.35/2.20 at 596/4,000
games, H0). This was a valid data-only continuation from `enyo-7.4.0-rc1`:
the corrected LC0 Test91 V6 result targets did not improve the selected parent
regimen. Candidate SHA-256: `b5ffc2ba93b917ac6d91c60b0b6908afdfc6f789154862b71fa59f6c4406a70d`.
Rejected: `enyo-7.5.0-rc9` (data-mixture variable from `enyo-7.4.0-rc1`:
2,000,000,000 selected-corpus records plus 666,833,785 deterministic Test91
records, 25.0047% Test91; manifest SHA-256
`53f8e1a29bebe4b0d7a81e023bf68cbe69e41a14d0ab9d9e164c23ed25583893`; -39.3
±15.5 Elo, LLR -2.43/2.20 at 754/4,000 games, H0).
Void: `enyo-7.5.0-rc10` (data-only continuation from `enyo-7.4.0-rc1` using
the preserved historical combined corpus, SHA-1
`6014cd9863ee31b02ab877adb606e0b4d14df70f`; inconclusive, -5.7 Elo, LLR
-1.79/2.20 — neither H0 nor H1 reached).

Reserved (recorded after launch, 2026-11-15): `enyo-18.0.0-rc1` on pwa-llm,
scratch root with no `continue_from` or `initialize_from`, approved by the
user in session. `architecture.json` is unchanged from Architecture 7; the
number 18 marks a new lineage root, which the naming rule does not cover, so
the name is open to correction. One 762,940-superbatch run (batch 16,384 x 8,
100B positions), LR 0.001 -> 0.000005 cosine, WDL 0.025, sfbinpack loader,
min ply 24, on `~/assets/training/binpacks/master-interleaved-v1.binpack`
(259,017,610,955 bytes: the eleven Stockfish master binpacks plus
test80-2024-03, interleaved chunk-wise by `tools/pack/interleave_binpacks.py`
with seed 1; raw scores, not recalibrated). Differs from the Enyo-7 path in
origin, corpus, batch size and schedule at once, so a result cannot be
attributed to one of them.

|  # | Selected run    | Commit   | Host     | Parent SPRT  | SF            | Change
|----|-----------------|----------|----------|--------------|---------------|--------------------------------
|  0 | enyo-5.0.0-rc1  | 11aa8bc4 | pwa-llm  | —            | -315.2 ± 10.6 | scratch SCReLU root
|  1 | enyo-5.1.0-rc2  | 715837c6 | pwa-5090 | +64.2 ± 15.9 | -263.1 ± 9.3  | shuffled self-play-1.2 corpus
|  2 | enyo-5.2.0-rc1  | d2da4c89 | pwa-llm  | +38.2 ± 12.6 | -254.2 ± 9.0  | next full dose
|  3 | enyo-5.3.0-rc2  | b6db90d0 | pwa-5090 | +28.7 ± 11.6 | -232.0 ± 8.5  | final LR 0.000005 → 0.000010
|  4 | enyo-5.4.0-rc1  | f09a933b | pwa-llm  | +8.8 ± 10.7  | -220.2 ± 8.1  | next full dose
|  5 | enyo-5.5.0-rc1  | bf950072 | pwa-llm  | +11.1 ± 10.6 | -216.1 ± 8.4  | next full dose
|  6 | enyo-5.6.0-rc2  | 074dc6b9 | pwa-5090 | +24.8 ± 10.4 | -204.8 ± 8.1  | initial LR 0.0010 → 0.0020
|  7 | enyo-5.7.0-rc2  | 73b965c3 | pwa-5090 | +19.0 ± 10.1 | -191.2 ± 8.1  | initial LR 0.0020 → 0.0030
|  8 | enyo-5.8.0-rc1  | e0ab687d | pwa-llm  | +13.0        | -192.6 ± 8.1  | next full dose
|  9 | enyo-5.9.0-rc6  | 0ec2e1fc | pwa-5090 | +7.6 ± 10.2  | -183.9 ± 7.7  | two corpus passes
| 10 | enyo-5.10.0-rc2 | 568b0c3c | pwa-5090 | +9.0 ± 10.7  | -185.3 ± 8.0  | four corpus passes
| 11 | enyo-6.0.0-rc1  | 486c6e59 | pwa-llm  | +2.8 ± 10.3  | -165.1 ± 7.4  | activated L2-to-output skip
| 12 | enyo-6.1.0-rc2  | 2741cfd1 | pwa-llm  | +7.9 ± 10.1  | -170.5 ± 7.7  | two corpus passes
| 13 | enyo-6.2.0-rc4  | 18dd99b0 | pwa-5090 | +2.1 ± 10.6  | -164.3 ± 7.5  | initial LR 0.0030 → 0.0040
| 14 | enyo-6.3.0-rc3  | 1aa663b7 | pwa-llm  | +12.0 ± 10.4 | -148.6 ± 7.5  | final LR 0.000010 → 0.000005; tie-break +6.6 ± 5.5 vs rc4
| 15 | enyo-6.4.0-rc1  | cd7728bc | pwa-llm  | +0.7 ± 10.2  | -156.1 ± 7.4  | three corpus passes; tie-break +4.4 ± 5.6 vs rc2
| 16 | enyo-7.0.0-rc2  | dde4b621 | pwa-llm  | +9.3          | -158.9 ± 7.4  | 32 input buckets; explicit initialization
| 17 | enyo-7.1.0-rc1  | fc024567 | pwa-llm  | +2.3 ± 9.9   | -147.2 ± 7.3  | full continuation
| 18 | enyo-7.2.0-rc1  | 296f72bd | pwa-llm  | +4.2          | -157.0 ± 7.6  | second full continuation
| 19 | enyo-7.1.0-rc2  | 22a4a975 | pwa-5090 | +5.7 ± 5.4    | -150.3 ± 7.4  | independent continuation trajectory
| 20 | enyo-7.2.0-rc2  | e383cbf4 | pwa-llm  | +2.0 ± 5.6    | -145.2 ± 7.3  | one additional corpus pass; SF tie-break over rc3
| 21 | enyo-7.3.0-rc3  | 0f7513e1 | pwa-llm  | +6.8 ± 6.1    | -138.9 ± 7.1  | recalibrated nodes5000pv2 labels
| 22 | enyo-7.4.0-rc1  | d7a94117 | pwa-llm  | +0.4 ± 5.7    | -138.0 ± 7.3  | WDL 0.05 → 0.025; SF tie-break
| 23 | enyo-7.5.0-rc14 | 2f02279d | pwa-llm  | +0.1 ± 6.6    | -127.1 ± 6.9  | SF tie-break vs actual champion rc1 (inconclusive); continue_from mislabeled as rc3 (naming collision), rejected -9.3±7.9 there — not a valid Parent SPRT chain link, see note above
| 24 | enyo-7.5.0-rc16 | 6034eea0 | pwa-llm  | +8.3 ± 6.7    | -124.9 ± 7.0  | next full dose off rc14, fresh 100M-position slice
| 25 | enyo-7.5.0-rc17 | c5840c2b | pwa-llm  | +2.9 ± 6.2    | -121.0 ± 6.8  | SF tie-break (inconclusive parent SPRT); next full dose off rc16, fresh 100M-position slice
| 26 | enyo-7.5.0-rc21 | d3162e81 | pwa-llm  | +11.5 ± 7.8   | -125.4 ± 6.8  | clean H1 win; rc18/rc19/rc20 all rejected on SF tie-break at intermediate slices
| 27 | enyo-7.5.0-rc27 | 11b768c3 | pwa-llm  | +3.7 ± 6.2    | -123.5 ± 6.9  | SF tie-break (inconclusive); rc22-rc26 all rejected first (lr/wdl probes reverted, data slices lost tie-break)
| 28 | enyo-7.5.0-rc36 | 95e1e0b5 | pwa-llm  | -2.9 ± 6.1    | -122.7 ± 6.5  | SF tie-break (inconclusive, narrow); rc28-rc35 all rejected first (sb sweep + data slices), rc27 independently verified +5.4 vs rc17 at 8000g first
| 29 | enyo-7.5.0-rc38 | c36a9bef | pwa-llm  | -4.9 ± 6.2    | -117.9 ± 6.7  | SF tie-break only (direct SPRT unfavorable); best SF in the chain; flagged for mechanical offset-incrementing without a real hypothesis (rc37/rc38); **REVERTED 2026-09-10**: 12,000-game wide-SPRT vs rc36 resolved -3.8 ±3.8, LLR -2.29/2.20 (H0); not a valid parent, reverted to rc36 (row 28)

Void: `enyo-10.0.0-rc1` (independent dense heads; no promotion).
Void: `enyo-11.0.0-rc1` (reset-tail output scale; invalid startpos +2023 cp).
Rejected: `enyo-11.0.0-rc2` (ordinary ReLU dense tail; -86.5 ±20.1, H0).
Rejected: `enyo-12.0.0-rc1` (16 input buckets; +0.4 ±5.5, SF tie-break −145.6 vs −138.0).
Rejected: `enyo-13.0.0-rc1` (unfactorised inputs; -11.1 ±8.0, H0).
Rejected: `enyo-14.0.0-rc1` (512-wide accumulator; -50.9 ±18.7, H0).



# Enyo 1 Lineage

`Parent SPRT` is historical and is not comparable to the Enyo-5/6 results.
Rows are the selected direct-weight path; omitted versions are rejected or
non-ancestor sibling experiments. Scale rows are deterministic export
transforms, not training runs.

|  # | Selected run                      | Commit   | Host | Parent SPRT   | SF | Change
|----|-----------------------------------|----------|------|---------------|----|--------------------------------
|  0 | enyo-scratch-broad-1.0.0-rc1      | 813e229b | —    | —             | —  | random Pylon root; 2.8B positions; 65,536 SB; LR 0.0010 → 0.000005
|  1 | enyo-scratch-long-1.0.0-rc1       | a0dc31b6 | —    | +93.2         | —  | random weight root; broad Bullet corpus; 196,608 SB
|  2 | enyo-scratch-long-1.1.0-rc1       | 7c7a0764 | —    | +45.9         | —  | 196,608 SB
|  3 | enyo-scratch-long-1.2.0-rc1       | a1c8368a | —    | +25.8         | —  | 196,608 SB
|  4 | enyo-scratch-long-1.3.0-rc1       | 1c22805e | —    | +16.0         | —  | 196,608 SB
|  5 | enyo-scratch-long-1.4.0-rc1       | d6c42e4c | —    | +7.4          | —  | 196,608 SB
|  6 | enyo-scratch-long-1.5.0-rc1       | 8cea17a7 | —    | +0.5          | —  | 98,304 SB
|  7 | enyo-1.0.0-rc1                    | 9d829ac0 | —    | +4.9          | —  | 98,304 SB
|  8 | enyo-1.1.0-rc2                    | f054980b | —    | +9.5          | —  | WDL 0.05 → 0.15; broad Bullet corpus; 98,304 SB
|  9 | enyo-1.2.0-rc1                    | 9efb7de5 | —    | +3.0          | —  | broad Bullet corpus; 98,304 SB
| 10 | enyo-1.3.0-rc2                    | 5572685c | —    | +13.4         | —  | broad Bullet corpus; 98,304 SB
| 11 | enyo-1.4.0-rc2                    | 15cc8afb | —    | +2.1          | —  | broad Bullet corpus; 98,304 SB
| 12 | enyo-1.5.0-rc2                    | 11a87ee5 | —    | +11.8         | —  | broad Bullet corpus; 32,768 SB
| 13 | enyo-1.6.0-rc2                    | 0e829bfc | —    | +13.9         | —  | broad Bullet corpus; 32,768 SB
| 14 | enyo-1.7.0-rc3                    | db3687ff | —    | +5.3          | —  | LR 0.00025; broad Bullet corpus; 32,768 SB
| 15 | enyo-1.11.0-rc2                   | 97757ff5 | —    | +7.7          | —  | LR 0.000125; broad Bullet corpus; 16,384 SB
| 16 | enyo-1.12.0-rc1                   | dd74d02b | —    | +6.7          | —  | broad Bullet corpus; 16,384 SB
| 17 | enyo-1.13.0-rc2                   | 236ed823 | —    | +7.9          | —  | FarseerT76; WDL 0.30; 256 SB
| 18 | enyo-1.14.0-rc2                   | 4eef094b | —    | +2.1          | —  | FarseerT76; WDL 0.05; 256 SB
| 19 | enyo-1.15.0-rc2                   | d09170e7 | —    | +12.3         | —  | FarseerT76; LR 0.0001; 256 SB
| 20 | enyo-1.16.0-rc3                   | 79cd05cb | —    | +4.9          | —  | T60T70wIsRightFarseer; 256 SB
| 21 | enyo-1.20.0-rc12                  | 586a5c29 | —    | +2.1          | —  | Pylon; output-only; 512 SB
| 22 | enyo-1.28.0-rc16                  | f37067d2 | —    | +10.2         | —  | Pylon; input-only; LR 0.00001; 256 SB
| 23 | enyo-1.30.0-rc3                   | —        | —    | +25.3 ± 13.9  | —  | output-head x0.48; grid: rc1 .52, rc2 .56, rc4 .50, rc5 .46
| 24 | enyo-1.30.0-rc3-unscaled          | —        | —    | —             | —  | inverse scale; input/L1/L2 identical across grid
| 25 | enyo-1.31.0-rc57                  | 3c464a3b | —    | +8.6          | —  | Stockfish-static relabel of 46.4M self-play positions; 354 SB
| 26 | enyo-1.32.0-rc10                  | 650a4fcc | —    | +3.3          | —  | recalibrated nodes5000pv2 labels; 4,000 SB

Reserved: `enyo-19.0.0-rc1` on pwa-llm (architecture 19: eval_scale 400 -> 160,
target sigmoid scale as in Grapheus/Berserk; initialize_from enyo-7.5.0-rc36 via the
scale-aware converter; otherwise rc36's regimen and data slice). Single variable.
Control at eval_scale 400 not yet reserved.

Reserved: `enyo-7.5.0-rc88` on pwa-llm (continue_from rc36; same regimen and data slice as rc36; single variable: sfbinpack max_abs_cp 10000 -> 300, from the structural audit of rc36 vs the Stockfish net).

Reserved: `enyo-20.0.0-rc1` on pwa-llm (architecture 20: input buckets 32 -> 16, output buckets 8 -> 1; initialize_from enyo-7.5.0-rc36 with averaged contraction; rc36 regimen and data slice).
Void: `enyo-20.0.0-rc1` (16 input buckets with 1 output bucket; failed at train: mixed activation requires the 8-bucket base).
Reserved: `enyo-20.0.0-rc2` on pwa-llm (16 input buckets, 8 output buckets; single variable vs rc36).
Reserved: `enyo-7.5.0-rc89` on pwa-llm (continue_from rc36; lr_schedule step, drop x0.3 every 256 superbatches; otherwise rc36's regimen and data slice. Single variable.)
Reserved: `enyo-7.5.0-rc90` on pwa-llm (continue_from rc36; focused fine-tune on the 11,042 positions where rc36 and Stockfish disagree by more than 300 cp; 40 superbatches, lr 0.001).
Reserved: `enyo-7.5.0-rc91` on pwa-llm (continue_from rc36; same disagreement set; gentler: lr 0.0002, 10 superbatches; follows rejected rc90).
Reserved: `enyo-7.5.0-rc92` on pwa-llm (continue_from rc36; rc90 regimen with phase-normalised targets; rc90 and rc91 trained on raw centipawns and were rejected).
Reserved: `enyo-7.5.0-rc93` on pwa-llm (control: continue_from rc36 on the normal Pylon data, lr 0.000001, 2 superbatches).
Reserved: `enyo-7.5.0-rc94` on pwa-llm (continue_from rc36; 10% disagreement positions + 90% normal Pylon, both phase-normalised; lr 0.001, 40 superbatches).
