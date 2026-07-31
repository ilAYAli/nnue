All development is done on pwa-5090 ~/code/cpp/chess/nnue
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
"run:" enyo-1.0.0-rc1
"continue_from": enyo-scratch-long-1.5.0-rc1

commit: 8cea17a7
"run": "enyo-scratch-long-1.5.0-rc1",
"continue_from": "enyo-scratch-long-1.4.0-rc1"

commit: d6c42e4c
"run": "enyo-scratch-long-1.4.0-rc1"
"continue_from": "enyo-scratch-long-1.3.0-rc1"

commit: 1c22805e
"enyo-scratch-long-1.3.0-rc1"
"continue_from": "enyo-scratch-long-1.2.0-rc1"

commit: a1c8368a
"run": "enyo-scratch-long-1.2.0-rc1"
"continue_from": "enyo-scratch-long-1.1.0-rc1"

commit: 7c7a0764
"run": "enyo-scratch-long-1.1.0-rc1",
"continue_from": "enyo-scratch-long-1.0.0-rc1"

commit: 5a6028ef
"run": "enyo-scratch-long-1.0.0-rc1"


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

Template file is left in place on pwa-5090 at
~/code/cpp/chess/forge/templates/selfplay-to-bullet-sf-oracle.template.json, uncommitted, NOT working.
Test artifacts left for inspection: ~/code/cpp/chess/nnue/data/selfplay/.tmp-sfo-template-test/ (20 PGN
files) and the Forge run directory
~/code/cpp/chess/forge/runs/selfplay-to-bullet-sf-oracle-.tmp-sfo-template-test-20260731-012513/.

Next step if picking this back up: debug the template's setup phase specifically (materialize_inputs,
the nnue source install step, or the bullet-utils cargo build) on a single worker directly before
attempting another fleet-wide deploy - the hang happened before any task produced output, so it's a
setup/deploy problem, not a labeling-script problem (the script itself is confirmed correct locally).
