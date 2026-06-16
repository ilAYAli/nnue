# Enyo NNUE Improvement Plan

Durable conclusions and the active hypothesis only. See `AGENTS.md` for
roles, rules, and validation order. Run-by-run logs belong in git history
and run directories, not in this file.

## Status

The `native-1.x`–`native-7.x` lineage is rejected pending re-test. A month
of iteration ran against a validation chain that did not prove the engine
was loading the candidate net: missing `nnue_file` silently fell back to
default/legacy behavior, no startup line confirmed evaluator type or file
hash, and static gates were treated as promotion evidence without verified
game results. The reported `uho52_lr1e4_e64.nn` -730 Elo result is
uninterpretable, not a measurement.

A new lineage, `native-8.0.0`, starts after the validation chain is
hardened in the engine (`~/code/cpp/chess/enyo`) and in
`./nnue-run doctor`. No training counts until then.

## Active Hypothesis

### Iteration 0 — pipeline parity (not a candidate)

The rebuilt validation chain reports a known-good result correctly. Train
a tiny config on a 100k-row slice of `data/bullet/lc0q10m.bullet` to
produce a loadable `.nn`. Run train/export/engine parity. Run a 50-game
SPRT vs `default.net` to verify both sides log the expected evaluator and
file hash. The candidate is expected to lose; the test is whether the
chain reports it correctly.

### Iteration 1 — `native-8.0.0-rc1`

Run only if iteration 0 passes.

Lc0 test91 Q-targets at 10M rows produce a native-MAJOR baseline that
holds its own against `default.net`. One variable family changed: data.
Architecture, objective, learning rate, and batch size come from the
cleanest pre-codex recipe under `recipes/`.

`changed_variables`:
- `source_bullet`: `data/bullet/lc0q10m.bullet`

Promotion criterion: 300-game smoke vs `default.net` neutral-positive
(LOS > 30%), followed by a longer SPRT. Per `AGENTS.md`, three
consecutive rejected candidates from the same family close the family.

## Closed Lanes

Do not restart without a new representation or objective:

- `native-1.x`–`native-7.x` as a lineage. Two representative nets
  (`uho52_lr1e4_e64.nn`, `enyo-native-7.6.0-rc1.nn`) are kept for a
  one-time post-fix re-test with the corrected `nnue_file=...default.net`
  form. The lineage is otherwise rejected and not a baseline.
- Static-only promotion. MAE/sign improvements on heldout data do not
  predict game strength.
- Repo-local SPRT wrappers. Game validation goes through forge/Crucible.
- Move-policy sidecar runtime path.
- Architecture or features that fail export parity, engine-static gate,
  or NPS before games.

## Required Sanity Test Form

Every candidate SPRT uses the explicit `nnue_file=` form on both sides.
The engine logs evaluator type, absolute `nnue_file` path, and the
SHA-256 of the loaded weights on startup; the SPRT runner asserts these
match the expected values before counting any game. Missing or invalid
`nnue_file` is a hard startup failure, never a silent fallback.

```sh
sprt \
  --reference ~/assets/engines/enyo_HASH \
  --reference-uci "nnue_file=$HOME/code/cpp/chess/enyo/net/default.net" \
  --candidate ~/assets/engines/enyo_HASH \
  --candidate-uci "nnue_file=/absolute/path/to/candidate.nn" \
  --games 300
```

## Pre-Training Gates

In order. Stop early on any failure. Do not rescue a failed family with
near-identical reruns.

1. `./nnue-run doctor -c build.json` passes, including the SPRT preflight
   that dry-runs both engines and asserts evaluator type and weight hash
   on both sides.
2. Training and export complete and produce one intended `.nn`.
3. Train/export/engine parity passes on the produced net within
   quantization tolerance.
4. 200–300 game smoke vs `default.net` is neutral-positive.
5. Longer SPRT only after smoke passes.

## Active Workflow

Use `./nnue-run` driven by `build.json`. The config states `hypothesis`,
`changed_variables`, ordered `stages`, and pass/fail `gates`. One commit
per iteration changing only `build.json` (and a deliberate update to this
file when the hypothesis line changes). Do not start a training run from
a dirty tree otherwise.

Prefer the Bullet/BulletLib path for training. Large data conversion or
mixing goes through compiled C++ tools called from `build.py`.
