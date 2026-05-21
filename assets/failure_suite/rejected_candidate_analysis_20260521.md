# Rejected Candidate Failure Analysis - 2026-05-21

Compared replay failure-suite CSVs from the recent rejected candidates.

## Candidate Summaries

| candidate | positions | better | worse | sum diff | worst | best |
|---|---:|---:|---:|---:|---:|---:|
| material_all | 913 | 64 | 55 | +561 | -563 | +545 |
| material_floathead | 913 | 70 | 74 | -31 | -563 | +416 |
| kingbucket_all | 913 | 68 | 54 | +1491 | -448 | +545 |
| hardcase_w12 | 913 | 67 | 58 | -117 | -511 | +400 |
| instability_w5 | 506 | 42 | 37 | +1384 | -338 | +512 |
| old_control | 506 | 35 | 32 | +1277 | -287 | +545 |

## Repeated Tail Regressions

These positions regressed in more than one independent rejected candidate:

- `caissa-x_vs_EnyoBot_0waEuGxc.log` ply `135`: 4 hits.
  Candidate moves varied, but the oracle/reference repeatedly preferred
  `a8a5`.
- `ArasanX_vs_EnyoBot_7pzXRQFZ.log` ply `63`: 2 hits.
  Material-head variants both chose `f7g7`; reference/oracle preferred `f8e7`.
- `Bot1nokk_vs_EnyoBot_mEIqToS1.log` ply `253`: 2 hits.
  Material-all and kingbucket-all both chose `b6a5`; reference/oracle preferred
  `a6a5`.
- `SF_Bot1nok_vs_EnyoBot_SbzD89kx.log` ply `101`: 2 hits.
  Material-all and kingbucket-all both chose `f5g5`; reference preferred
  `g6f7`, oracle preferred `g6h7`.
- `ArasanX_vs_EnyoBot_Q0a5I19l.log` ply `79`: 2 hits.
  Float-head and hardcase both chose `e4h4`; reference/oracle preferred
  `a4b4`.
- `caissa-x_vs_EnyoBot_0waEuGxc.log` ply `91`: 2 hits.
  Material-all and kingbucket-all both chose `g5f3`; reference/oracle preferred
  rook moves.
- `EnyoBot_vs_Lynx_BOT_jjThVRPN.log` ply `112`: 2 hits.
  Material-all and kingbucket-all both chose `g4g5`; reference preferred
  `a3b4`, oracle preferred `a3b2`.
- `stage270_vs_EnyoBot_kp3inZBb.log` ply `123`: 2 hits.
  Float-head and kingbucket-all both chose `h4g3`; reference/oracle preferred
  `h4h3`.

## Interpretation

The failures are not random single-candidate noise. Several independent
training directions improve aggregate score while creating the same tail
regressions. That means the next useful work is not another bulk scalar-eval
candidate on the same data. The next training signal should be move-choice
aware: isolate these repeated tail positions, classify them, and build a gate or
training target that directly penalizes moving away from the oracle/reference
choice.
