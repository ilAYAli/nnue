All development is done on pwa-llm ~/code/chess/nnue
read AGENTS.md there and the associated SKILL.md's

The best net is currently ~ -154 elo from SF and ~ -122 from Berserk:

enyo-1.30.0-rc3.nn vs berserk-9b84c340af7e.nn (1500 games):
elo=-122.2  llr=-2.29/2.20 (-104%)  los=0.0%  ci=31.9  draw=35.4%

enyo-1.30.0-rc3.nn vs nn-0ee0657fb25e.nnue (1500 games):
elo=-154.4  llr=-22.66/690.78 (-3%)  los=0.0%  ci=15.4  draw=32.9%


Update: the currently best net is always ~/assts/nets/reference.net (enyo-1.32.0-rc10.nn)

The NNUE architecture is failry close to Berserk, but I have been able to close the gap further.
I have tried different architectures, selfplay, stockfish binpacks, and additional features such
as FullThreats.

Look at the git log to investigate the different lineages.

I am contemplating re-doing an earlier experiment and have a competition between multiple architecutes and/or features, trained without using existing tensors.


*** This is the enyo-1 lineage ***

commit: 9d829ac0
Elo 4.9,LLR 0.18/2.20 (8%)
"run:" enyo-1.0.0-rc1
"continue_from": enyo-scratch-long-1.5.0-rc1

commit: 8cea17a7
Elo 0.5,LLR 0.02/2.20 (1%)
"run": "enyo-scratch-long-1.5.0-rc1",
"continue_from": "enyo-scratch-long-1.4.0-rc1"

commit: d6c42e4c
"run": "enyo-scratch-long-1.4.0-rc1"
"continue_from": "enyo-scratch-long-1.3.0-rc1"

commit: 1c22805e
Elo 7.4,LLR 0.35/2.20 (16%)
"enyo-scratch-long-1.3.0-rc1"
"continue_from": "enyo-scratch-long-1.2.0-rc1"

commit: a1c8368a
Elo 25.8,LLR 1.17/2.20 (53%)
"run": "enyo-scratch-long-1.2.0-rc1"
"continue_from": "enyo-scratch-long-1.1.0-rc1"

commit: 7c7a0764
"run": "enyo-scratch-long-1.1.0-rc1",
"continue_from": "enyo-scratch-long-1.0.0-rc1"

commit: 5a6028ef
Elo 93.2,LLR 2.26/2.20 (103%)
"run": "enyo-scratch-long-1.0.0-rc1"

  ┌──────────┬─────────────────────────────┬─────────────────────────────┬───────┬──────────────────┐
  │  commit  │             run             │        continue_from        │  Elo  │       LLR        │
  ├──────────┼─────────────────────────────┼─────────────────────────────┼───────┼──────────────────┤
  │ 9d829ac0 │ enyo-1.0.0-rc1              │ enyo-scratch-long-1.5.0-rc1 │ +4.9  │ 0.18/2.20 (8%)   │
  ├──────────┼─────────────────────────────┼─────────────────────────────┼───────┼──────────────────┤
  │ 8cea17a7 │ enyo-scratch-long-1.5.0-rc1 │ enyo-scratch-long-1.4.0-rc1 │ +0.5  │ 0.02/2.20 (1%)   │
  ├──────────┼─────────────────────────────┼─────────────────────────────┼───────┼──────────────────┤
  │ d6c42e4c │ enyo-scratch-long-1.4.0-rc1 │ enyo-scratch-long-1.3.0-rc1 │ +7.4  │ 0.35/2.20 (16%)  │
  ├──────────┼─────────────────────────────┼─────────────────────────────┼───────┼──────────────────┤
  │ 1c22805e │ enyo-scratch-long-1.3.0-rc1 │ enyo-scratch-long-1.2.0-rc1 │ +16.0 │ 0.71/2.20 (32%)  │
  ├──────────┼─────────────────────────────┼─────────────────────────────┼───────┼──────────────────┤
  │ a1c8368a │ enyo-scratch-long-1.2.0-rc1 │ enyo-scratch-long-1.1.0-rc1 │ +25.8 │ 1.17/2.20 (53%)  │
  ├──────────┼─────────────────────────────┼─────────────────────────────┼───────┼──────────────────┤
  │ 7c7a0764 │ enyo-scratch-long-1.1.0-rc1 │ enyo-scratch-long-1.0.0-rc1 │ +45.9 │ 2.12/2.20 (96%)  │
  ├──────────┼─────────────────────────────┼─────────────────────────────┼───────┼──────────────────┤
  │ 5a6028ef │ enyo-scratch-long-1.0.0-rc1 │ (scratch, no continue_from) │ +93.2 │ 2.26/2.20 (103%) │
  └──────────┴─────────────────────────────┴─────────────────────────────┴───────┴──────────────────┘

Near-tip chain (enyo-1.30.0-rc3 forward to the current champion) - verified separately,
2026-08-02. This does NOT connect to the table above: enyo-1.30.0-rc3 itself has no
recoverable commit anywhere in history (checked all commits' build.json content, not just
subject-line grep) and the net file no longer exists on disk. Versions 1.1.0-1.29.0 were not
traced since there is no endpoint to connect them to.

  ┌──────────┬────────────────────────────────┬───────────────────────────────┬───────┬─────────────────────────┐
  │  commit  │              run               │         continue_from         │  Elo  │           LLR           │
  ├──────────┼────────────────────────────────┼───────────────────────────────┼───────┼─────────────────────────┤
  │    ?     │ enyo-1.30.0-rc3                │ (untraceable - no commit,     │   -   │            -            │
  │          │                                │  no net file on disk)         │       │                         │
  ├──────────┼────────────────────────────────┼───────────────────────────────┼───────┼─────────────────────────┤
  │    -     │ enyo-1.30.0-rc3-unscaled       │ enyo-1.30.0-rc3               │  n/a  │ not an independent run  │
  │          │ (manual output-scale fix on    │                               │       │ (never SPRT-tested or   │
  │          │  rc3's own weights)            │                               │       │  committed)             │
  ├──────────┼────────────────────────────────┼───────────────────────────────┼───────┼─────────────────────────┤
  │ 3c464a3b │ enyo-1.31.0-rc57               │ enyo-1.30.0-rc3-unscaled      │ +8.6  │ 0.37/2.20 (17%)         │
  ├──────────┼────────────────────────────────┼───────────────────────────────┼───────┼─────────────────────────┤
  │ 650a4fcc │ enyo-1.32.0-rc10               │ enyo-1.31.0-rc57              │ +3.3  │ los=81.4%, ci=7.3       │
  │          │ (current champion)             │                               │       │                         │
  └──────────┴────────────────────────────────┴───────────────────────────────┴───────┴─────────────────────────┘

Naming collision: the name enyo-1.32.0-rc10 was used twice. An earlier, unrelated attempt was
rejected on 2026-07-15 (commit 7e31912f, continue_from enyo-1.31.0-rc42, elo=-69.6) - a
different net that happened to reuse the same name. Only commit 650a4fcc (2026-07-29, above)
is the actual current champion file.

Rejected side-branches off the current tip (2026-08-01/02, this session, not part of the
accepted chain above):
  enyo-1.33.0-rc9  (continue_from rc10, data/selfplay/gen4/gen4-sf-oracle.bullet - self-play
                    generated with Stockfish's own net loaded into the search engine, i.e. the
                    wrong policy net for self-distillation): elo=-17.3  llr=-2.40/2.20 (-109%)
                    (1708/6000 games) - rejected.
  enyo-1.33.0-rc10 (continue_from rc10, data/selfplay/gen5/gen5-sf-oracle.bullet - corrected
                    recipe, rc10 plays itself, genuine SF static-eval relabel, matching rc57's
                    actual recipe): elo=-9.0  llr=-2.25/2.20 (-102%) (3014/6000 games) - rejected.
  Conclusion: even the corrected self-play recipe does not reproduce rc57's gain against the
  now-mature rc10 - the incremental self-play-fine-tune path off rc10 appears saturated.

*** default.net reference ***

default.net (used below) is ~/code/rice/src/hexadecane_512_v2.net on pwa-mbp (with 4 bytes appended, IIRC) -
a DIFFERENT reference engine than nn-0ee0657fb25e.nnue used everywhere else in this file. Elo numbers
against the two are not directly comparable.

*** enyo-1 lineage elo progression (vs default.net, benchmarks/default-net.jsonl) ***

enyo-scratch-broad-1.0.0-rc1   -236.0  (1000 games, 2026-07-03)
enyo-scratch-long-1.0.0-rc1    -169.3  (500 games,  2026-07-05)
enyo-scratch-long-1.1.0-rc1    -126.2  (500 games,  2026-07-05)
enyo-scratch-long-1.3.0-rc1    -113.7  (500 games,  2026-07-06)
(1.2.0, 1.4.0, 1.5.0 not present in benchmarks/default-net.jsonl)

enyo-1.0.0-rc1 (continue_from enyo-scratch-long-1.5.0-rc1) was the first net of this chain ever
benchmarked against Stockfish directly:
elo=-181.2  llr=-17.52/690.78  los=0.0%  ci=29.3  (1000 games, 2026-07-07, benchmarks/stockfish-net.jsonl:1)

*** 2026-07-30 reconstruction ***

enyo-ancestor.1.0.0-rc1: reconstructed enyo-scratch-long-1.0.0-rc1's architecture/regimen from scratch
(196608 superbatches, wdl 0.05, min_ply 24, no continue_from) since the original
enyo-scratch-broad-1.0.0-rc1.bullet corpus no longer exists and its provenance is untraceable.
Substituted lc0-static + nodes5000pv2-recalibrated, both reconverted/relabeled at min_ply 24.
Result vs SF: elo=-208.6  llr=-19.24/690.78 (-3%)  los=0.0%  (1500 games, 2026-07-30).
~27 Elo below enyo-1.0.0-rc1's -181.2, same ballpark - the historical regimen isn't broken, the gap
is plausibly the data substitution. See benchmarks/stockfish-net.jsonl and commit e4e32ea.

Separately, tonight's own fresh 2-round scratch bootstrap (enyo-1.33.0-rc3 -> rc4, min_ply 16, same
combined corpus, 76762 superbatches/round, continuing from the prior round's own checkpoint each time)
reached elo=-180.6 vs SF after just 2 rounds (~153524 superbatches total) - matching enyo-1.0.0-rc1's
-181.2 starting point almost exactly, with LESS total dose than the single 196608-superbatch
enyo-ancestor.1.0.0-rc1 run above, which landed worse (-208.6). This suggests multiple shorter rounds
(fresh lr schedule reset each round) generalizes better than one long monolithic run of equal-or-greater
total dose - consistent with how the real historical chain was structured (7 successive rounds, not one).
Both rc3 and rc4 were rejected as promotion candidates against enyo-1.32.0-rc10 (the bar is much higher
now than enyo-1.0.0-rc1's original starting point), which is expected and not evidence against the
approach itself.

*** orphaned candidates (2026-07-30) ***

enyo-1.33.0-rc5.nn: trained and exported (continue_from rc10, lr 1e-5, farseerT74-recalibrated-600m.bullet,
4000 superbatches) but never benchmarked. A retroactive SPRT vs SF was started and then deliberately stopped
before completion - farseerT74-recalibrated-600m.bullet uses the same calibration curve already flagged
as likely mis-calibrated for this source, so spending a full 1500-game run on a presumed-bad candidate
wasn't worth it. No Elo vs SF exists for this candidate.

enyo-1.33.0-rc6: continue_from rc4, 196608 superbatches on lc0-nodes5000pv2-2b.bullet (min_ply 16) - started,
stopped almost immediately (only checkpoints/native-0 exists, no export) before the pivot to the
enyo-ancestor.1.0.0-rc1 reconstruction work. No net was ever produced, so no Elo vs SF exists for this
candidate.

*** correction (2026-07-30, later same day) ***

The rc57-vs-rc10 poison investigation above was resolved by clean re-benchmarks: both nets tested vs SF
with matched conditions (current engine enyo_11ca4d7, 4000 games each, tight bounds elo0=0/elo1=10/
alpha=beta=1e-300):

enyo-1.31.0-rc57.nn: elo=-146.7  ci=21.0  (4000/4000 games)
enyo-1.32.0-rc10.nn: elo=-139.8  ci=19.2  (4000/4000 games)

Statistically indistinguishable, rc10 if anything very slightly ahead. The original ~38 Elo gap
(-145.5 partial/invalid vs -183.5 at 500 games/older engine) that suggested rc10 had regressed from its
own parent was a measurement artifact - small sample size and an engine-binary mismatch between the two
original readings, not a real continue_from-chain regression. rc10 remains a perfectly good current
lineage tip; no re-basing needed. The best net vs SF currently on record is rc10/rc57, both ~-140 to -147.

*** self-play relabeling consistency investigation (2026-07-31 overnight) ***

Investigated whether data/selfplay/gen1/gen2/gen3 use consistent labeling before combining them.
Found they don't: gen1-live-sfo/gen1-sf-oracle.bullet (rc57's winning data) is SF-oracle static-eval
only, no tablebase correction. gen2.bullet and gen3/selfplay-sf-oracle-tb.bullet are both SF-oracle +
TB-corrected (manifest for gen2.bullet traces to a run literally named tb-sf-oracle-relabel-...).

Ran enyo-1.33.0-rc8 (gen3's non-TB variant selfplay-sf-oracle.bullet, continue_from rc10, same regimen
as the rejected rc7) as a single-variable isolation test. Result: elo=-2.7 ci=7.6 los=24.3% (6000/6000
games, full sample, no early termination) - not promoted (los well below the 75% threshold), but a large
improvement over rc7's clear rejection (elo=-23.1, los=6.1%). Consistent with (not proof of) tablebase
correction distorting calibration - see the rc7 residual_gate note above about static-metrics-improve-
but-play-doesn't.

Attempted to build a Forge template (selfplay-to-bullet-sf-oracle.template.json, in the forge repo,
NOT committed) to distribute tools/bullet/selfplay_to_bullet_sf_oracle.py across the fleet for
relabeling gen2 consistently (gen2 is the one generation with raw PGNs still on disk - gen1/gen3's
raw games were already cleaned up after their original conversion). Verified the underlying script
and its exact default parameters (skip_plies=8, min_depth=1, max_abs_cp=10000, mode=static,
tb_pieces=0, sf_net=nn-0ee0657fb25e.nnue) work correctly on a small local (non-distributed) sample.

The distributed Forge deployment of the new template hung: a 20-PGN-file test run
(selfplay-to-bullet-sf-oracle-.tmp-sfo-template-test-20260731-012513) sat at 0/20 done for ~3.7 hours
before I caught it and force-stopped it (forge stop --force: released=7 killed=48 reset=7). No task
logs were written at all, meaning tasks likely never got past setup (venv/cargo build install steps)
rather than hanging mid-relabel - root cause NOT diagnosed. Did not attempt a second blind distributed
run given the fleet-time cost of the first attempt (~4 hours across dozens of workers).

Template file is left in place on pwa-llm at
~/code/chess/forge/templates/selfplay-to-bullet-sf-oracle.template.json, uncommitted, NOT working.
Test artifacts left for inspection: ~/code/chess/nnue/data/selfplay/.tmp-sfo-template-test/ (20 PGN
files) and the Forge run directory
~/code/chess/forge/runs/selfplay-to-bullet-sf-oracle-.tmp-sfo-template-test-20260731-012513/.

Next step if picking this back up: debug the template's setup phase specifically (materialize_inputs,
the nnue source install step, or the bullet-utils cargo build) on a single worker directly before
attempting another fleet-wide deploy - the hang happened before any task produced output, so it's a
setup/deploy problem, not a labeling-script problem (the script itself is confirmed correct locally).

*** update: root cause partially found (2026-07-31, later) ***

Found and fixed one real bug in selfplay-to-bullet-sf-oracle.template.json: the nnue_repo include list
was missing tools/score/** (relabel_with_stockfish.py imports label_with_uci from there), causing
ModuleNotFoundError on every worker since only the included subset gets materialized - confirmed via
manual single-host reproduction (ssh + exact task command), fixed, and manually re-verified working
end-to-end on one worker (2156 positions converted and validated correctly, ~470 pos/sec).

Redeploying via Forge after the fix still did not complete a 20-PGN-file test (0/20 done after several
minutes). Diagnosed further: the PGN input directory was never materialized on the worker I checked
(~/.cache/forge/inputs/selfplay/ did not contain this test's cache key, despite the run showing tasks
as claimed/running). This is the materialize_inputs mechanism itself (core Forge behaviour, not
something a template JSON can fix) failing to complete/sync for this template, even for a handful of
small files. Stopped again (forge stop --force) rather than keep debugging Forge core solo - that's
outside what a template-only fix can address and is the other Claude session's remit (Forge core).

Net state: the template's own bug (missing include) is fixed and should be preserved if anyone picks
this back up, but there's a second, deeper issue in the input-materialization path that needs either
Forge-side debugging or the user's input before attempting gen2's real relabeling job again. Do not
retry the full job blind - both attempts so far cost real fleet time (~4h then ~15min) for zero
completed output.
