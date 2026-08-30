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
2. Long-running GPU training may run on `pwa-llm` or `pwa-5090`.
3. All commands should be executed with a trailing `;notifai-write "command completed"`
and you should immediately continue working when you receive this notification.
3. All servers you work on must have a tmux session names `nnue_cmd`
4. All commands must be executed in the `nnue_cmd` session.
5. Never create, rename, or interrupt tmux windows.
6. Never duplicate or interfere with active Forge jobs.

## Configuration

7. Never use foreign NNUE weights
8. `build.json` must not restate defaults or define undeclared parameters.
8. The only tracked diff during training should be `build.json`.
   Exceptions to this rule are needed changes for the build/iteration to succeed (nnue, Bullet, ...)
or explicitly approved architecture changes specified in `architecture.json`
9. Every run uses exactly one of `continue_from` or `initialize_from`; only an explicitly approved lineage root may omit both.
10. `initialize_from` should only be used for a compatible architecture change that preserves the parent information
11. `architecture.json` changes to this file requires explicit approval.

## Run naming

12. Format: `enyo-{architecture_number}.{promotion_number}.0-rc{iteration}`.
  `architecture_number` increments on an `architecture.json` change, resetting `promotion_number` to 0 and `iteration` to 1.
  `promotion_number` increments on acceptance, resetting `iteration` to 1.
  `iteration` is the candidate number within a promotion.
13. Put descriptions in `hypothesis`, never in the run name.
14. Never invent a naming scheme; ask first if the reservation flow does not fit.
15. Reserve every run name and host in `LINEAGE.md` before launch.
16. Run names that have no historical value can be reused.


## Experiment contract
17. Read `IMPROVEMENT_PLAN.md` before selecting an experiment.
18. Change exactly one meaningful variable per iteration.
19. `build.json` must contain a concise hypothesis explaining **why**.
20. During training, no source file may be modified except `build.json`, unless the iteration is an architecture experiment or requires a build-critical fix (a change required for that iteration to build or run).
21. Commit unrelated changes individually; never bundle them into the same commit as the active iteration's tracked diff. A build-critical source change required for an iteration belongs in that iteration's commit with `build.json` (and `architecture.json` when applicable). A semantic trainer change is its own experiment and must not be combined with a corpus or hyperparameter change.
22. Training/testing should always be active; never stop unless user interaction is required.
23. Scratch roots benchmark against `nn-1a298aa575a0.nnue`;
SPRT should be run like this (values should be changed to reflect the test):
```
HOOK_EVENTS=done,fail forge run sprt \
  --comment "enyo-1.32.0-rc10 vs nn-1a298aa575a0.nnue" \
  --candidate ~/assets/engines/reference \
  --reference ~/assets/engines/reference \
  --reference-net ~/assets/nets/nn-1a298aa575a0.nnue \
  --candidate-net ~/assets/nets/enyo-scc-1.0.0-rc1.nn \
  --elo0 0 --elo1 10 --alpha 1e-300 --beta 1e-300 \
  --games 4000; rc=$?; notifai-write "Forge SPRT completed rc=$rc"
```
24. Integrity gates (export, distinct-net, engine-load, start-position, catastrophic static) must pass before promotion; residual improvement is report-only.
25. Generated runs, caches, datasets, and validation output must never remain as source changes.

## Events & launch

26. NNUE completion is event-driven: `done`/`fail` arrives automatically via `llmsh`; never poll or arm background waiters.
27. Never launch a duplicate while a run is in flight.
28. For a long-running command that does not support `HOOK_EVENTS`, run `command; notifai-write.sh "command completed"` in `nnue_cmd` so completion is event-driven.
30. Keep `AUTO_ADVANCE` disabled unless explicitly requested for a single-host run.
31. On rejection, pick one new hypothesis; on acceptance, advance the
    data slice.
32. After a promotion is selected, record it in `LINEAGE.md`.

## Validation

33. Report games, Elo, confidence interval, LLR, LOS, draw rate, failures, and test conditions.
34. Compare candidates only under identical engines, books, time controls, and worker conditions.
35. Reject invalid exports, duplicate nets, engine-load failures, and catastrophic static failures.
36. Do not select a parent until every parallel candidate from the same parent has completed, failed, or been voided.
37. Preserve reproducibility evidence; never delete the only recorded copy of a result.

## Git

38. Work directly on `main`; never create branches.
39. Never pull on `pwa-llm`; integrate by fetch + cherry-pick.
40. Avoid fixup commits; amend when practical.
41. Never add AI or bot co-author trailers.
42. Stage only requested files and verify commit identity.
43. Include SPRT Elo in the commit subject whenever a result exists.
44. Never modify other repositories.
45. Before a `pwa-5090` winner becomes a parent, transfer the entire `runs/{run}/` directory, verify the checkpoint SHA-256, and confirm `continue_from` resolves to the optimizer checkpoint rather than the exported-net fallback.
46. After canonical promotion, update `candidate.net` and the Forge reference net; never point either at an active, rejected, or foreign net.
