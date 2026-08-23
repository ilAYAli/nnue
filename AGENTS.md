# Agent Rules

Repository-wide standing conventions. These rules always apply. Specialized workflows belong in `SKILL.md`.

Answer questions directly and use read-only inspection as needed. When pursuing an
explicitly requested experiment, edit `build.json` and `architecture.json` as the
experiment requires, and perform the normal Forge and Git workflow needed to run,
record, and promote it. Do not make unrelated changes.

## Repository

- `defaults.json` — complete shared training configuration.
- `build.json` — active run; only values that differ from `defaults.json`.
- `architecture.json` — trainer/export/runtime contract.
- `LINEAGE.md` — canonical run-name and parent registry.

## Hosts & Forge

1. `pwa-llm` is the only Forge coordinator.
2. Long-running training may run on `pwa-llm` or `pwa-5090`.
3. Use only the existing `nnue_cmd` tmux session.
4. Never create, rename, or interrupt tmux windows.
5. Never duplicate or interfere with active Forge jobs.

## Configuration

6. Never use foreign NNUE weights; Stockfish-generated data is allowed.
`build.json` must not restate defaults or define undeclared parameters.
7. The only tracked diff during training should be `build.json`.
   Exceptions to this rule are `architecture.json` (for architecture experiments), or
   other changes that are needed for the build/iteration to succeed (Bullet, ...).
8. Every run uses exactly one of `continue_from` or `initialize_from`; only a lineage root, including an explicitly approved new-architecture scratch root, may omit both.
9. `initialize_from` is valid only when the conversion preserves the parent's information; additive changes that append new input rows are valid, while changes that reshape an existing dimension require a new scratch root.
10. `architecture.json` changes only as a dedicated architecture experiment.

## Run naming

11. Format: `enyo-{architecture_number}.{promotion_number}.0-rc{iteration}`.
  `architecture_number` increments on an `architecture.json` change, resetting `promotion_number` to 0 and `iteration` to 1.
  `promotion_number` increments on acceptance, resetting `iteration` to 1.
  `iteration` is the candidate number within a promotion.
12. Put descriptions in `hypothesis`, never in the run name.
13. Never invent a naming scheme; ask first if the reservation flow does not fit.
14. Reserve every run name and host in `LINEAGE.md` before launch.
15. Run names that have no historical value can be reused.

## Experiment contract

16. Read `IMPROVEMENT_PLAN.md` before selecting an experiment.
17. Change exactly one meaningful variable per iteration.
18. `build.json` must contain a concise hypothesis explaining **why**.
19. During training, no source file may be modified except `build.json`, unless the iteration is an architecture experiment or requires a build-critical fix (a change required for that iteration to build or run).
20. Commit unrelated changes individually; never bundle them into the same commit as the active iteration's tracked diff. A build-critical source change required for an iteration belongs in that iteration's commit with `build.json` (and `architecture.json` when applicable). A semantic trainer change is its own experiment and must not be combined with a corpus or hyperparameter change.
21. Training/testing should always be active; never stop unless user interaction is required.
22. Scratch roots benchmark against `nn-0ee0657fb25e.nnue`; descendants use parent-relative SPRT with a 4,000-game cap. H0 rejects; H1 selects; if neither is reached by the cap, select by fixed-SF Elo under identical conditions.
23. Integrity gates (export, distinct-net, engine-load, start-position, catastrophic static) must pass before promotion; residual improvement is report-only.
24. Generated runs, caches, datasets, and validation output must never remain as source changes.

## Events & launch

25. NNUE completion is event-driven: `done`/`fail` arrives automatically via `llmsh`; never poll or arm background waiters.
26. Never launch a duplicate while a run is in flight.
27. For a long-running command that does not support `HOOK_EVENTS`, run `command; notifai-write.sh "command completed"` in `nnue_cmd` so completion is event-driven.
28. Launch exactly:

cd ~/code/chess/nnue
HOOK_EVENTS=done,fail MIN_SLOPE=0.05 SKIP_SMOKE=1 GAMES=4000 ./nnue iterate

29. Keep `AUTO_ADVANCE` disabled unless explicitly requested for a single-host run.
30. On rejection, pick one new hypothesis; on acceptance, advance the
    data slice.
31. After a promotion is selected, record it in `LINEAGE.md`.

## Validation

32. Report games, Elo, confidence interval, LLR, LOS, draw rate, failures, and test conditions.
33. Compare candidates only under identical engines, books, time controls, and worker conditions.
34. Reject invalid exports, duplicate nets, engine-load failures, and catastrophic static failures.
35. Do not select a parent until every parallel candidate from the same parent has completed, failed, or been voided.
36. Preserve reproducibility evidence; never delete the only recorded copy of a result.

## Git

37. Work directly on `main`; never create branches.
38. Never pull on `pwa-llm`; integrate by fetch + cherry-pick.
39. Avoid fixup commits; amend when practical.
40. Never add AI or bot co-author trailers.
41. Stage only requested files and verify commit identity.
42. Include SPRT Elo in the commit subject whenever a result exists.
43. Never modify other repositories.
44. Before a `pwa-5090` winner becomes a parent, transfer the entire `runs/{run}/` directory, verify the checkpoint SHA-256, and confirm `continue_from` resolves to the optimizer checkpoint rather than the exported-net fallback.
45. After canonical promotion, update `candidate.net` and the Forge reference net; never point either at an active, rejected, or foreign net.
