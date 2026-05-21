# Repeated Tail Target Taxonomy - 2026-05-21

Source: `repeated_tail_targets_20260521.csv`.

Legal-move scores: `repeated_tail_move_scores_20260521.csv`.

These are positions where at least two independent rejected candidates lost at
least `100cp` against the reference/oracle replay gate. The goal is not to
promote any candidate from these positions directly. The goal is to identify
which move-choice failures repeat across unrelated training attempts.

## Summary

- Targets: 13.
- Worst repeated regression: `-563cp`.
- Dominant pattern: low-material or queen-heavy conversion/defense positions.
- Not a normal opening or middlegame data-coverage problem.
- Next useful signal: move ranking / child-score targets around these positions,
  not another bulk scalar-eval run.

## Initial Categories

| id | log | ply | worst | category | notes |
|---:|---|---:|---:|---|---|
| 1 | `ArasanX_vs_EnyoBot_7pzXRQFZ.log` | 63 | -563 | queen/rook tactical defense | Candidates move `f7g7`; oracle/reference choose `f8e7`. |
| 2 | `Bot1nokk_vs_EnyoBot_mEIqToS1.log` | 253 | -448 | tablebase-adjacent conversion | `KQK+p`; candidates move king instead of pushing the pawn. |
| 3 | `caissa-x_vs_EnyoBot_0waEuGxc.log` | 135 | -296 | queen/minor king net | Six candidates miss `a8a5`. |
| 4 | `SF_Bot1nok_vs_EnyoBot_SbzD89kx.log` | 101 | -294 | low-material queen-vs-rook defense | Candidates play `f5g5`; oracle prefers king defense. |
| 5 | `ArasanX_vs_EnyoBot_Q0a5I19l.log` | 79 | -287 | queen/rook tactical defense | Candidates leave the queen active instead of `a4b4`. |
| 6 | `caissa-x_vs_EnyoBot_0waEuGxc.log` | 97 | -209 | blocked queen/rook maneuvering | Candidates miss `c7b7`. |
| 7 | `caissa-x_vs_EnyoBot_0waEuGxc.log` | 133 | -202 | queen/minor king net | Reference/oracle disagree on exact move, but candidates choose king moves. |
| 8 | `caissa-x_vs_EnyoBot_0waEuGxc.log` | 91 | -170 | blocked queen/rook maneuvering | Candidates choose knight capture; oracle/reference choose rook moves. |
| 9 | `Hypersion_vs_EnyoBot_npmgxvIO.log` | 97 | -168 | queen/rook conversion | Candidates overuse rook/queen checking moves. |
| 10 | `caissa-x_vs_EnyoBot_0waEuGxc.log` | 95 | -160 | blocked queen/rook maneuvering | Repeated miss around the same locked pawn structure. |
| 11 | `EnyoBot_vs_DarkOnBot_pm9SUqFN.log` | 138 | -143 | minor-piece pawn endgame | Candidates move bishop; oracle/reference move king. |
| 12 | `EnyoBot_vs_Lynx_BOT_jjThVRPN.log` | 112 | -135 | rook/minor pawn defense | Candidates push `g4g5`; oracle/reference move king. |
| 13 | `stage270_vs_EnyoBot_kp3inZBb.log` | 123 | -127 | rook/bishop pawn conversion | Candidates miss the direct pawn push `h4h3`. |

## Implication

The repeated tails are concentrated in conversion/defensive move choice. Bulk
Huber/MPE training can improve aggregate eval and still damage these positions.

The next candidate should therefore start from a small move-choice dataset:

1. For each target, evaluate all legal moves with the oracle.
2. Store top move, top-3 moves, and child score gaps.
3. Train or gate on move ranking consistency before any SPRT.

The legal-move score file is now available. It should be treated as a gate and
diagnostic input first; training from only 13 targets would overfit.
