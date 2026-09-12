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
3. All commands should be executed with a trailing `;notifai-write.sh "command completed"`
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
23. All build.json changes should be valudated with SPRT
# Candidate vs its continue_from/parent net; early H0/H1 stopping enabled
```
HOOK_EVENTS=fail,done GAMES=10000 \
  ./tools/validate/sprt_net.py \
  --early-stop \
  --candidate ~/assets/nets/enyo-7.5.0-rc47.nn \
  --reference ~/assets/nets/enyo-7.5.0-rc36.nn
```
# If the former is inconclusive, validate the candidate against Stockfish; run the full length

```
HOOK_EVENTS=fail,done MIN_SLOPE=0.05 SKIP_SMOKE=1 GAMES=4000 \
  ./tools/validate/sprt_net.py \
  --candidate ~/assets/nets/enyo-7.5.0-rc47.nn \
  --reference ~/assets/nets/nn-1a298aa575a0.nnue
```


24. Integrity gates (export, distinct-net, engine-load, start-position, catastrophic static) must pass before promotion; residual improvement is report-only.
25. SPRT vs. Stockfish is only needed if SRPT vs. the previous champion is inconclusive
26. Generated runs, caches, datasets, and validation output must never remain as source changes.

## Events & launch

27. NNUE completion is event-driven: `done`/`fail` arrives automatically via `llmsh`; never poll or arm background waiters.
28. Never launch a duplicate while a run is in flight.
29. For a long-running command that does not support `HOOK_EVENTS`, run `command; notifai-write.sh "command completed"` in `nnue_cmd` so completion is event-driven.
31. Keep `AUTO_ADVANCE` disabled unless explicitly requested for a single-host run.
32. On rejection, pick one new hypothesis; on acceptance, advance the
    data slice.
33. After a promotion is selected, record it in `LINEAGE.md`.

## Validation

34. Report games, Elo, confidence interval, LLR, LOS, draw rate, failures, and test conditions.
35. Compare candidates only under identical engines, books, time controls, and worker conditions.
36. Reject invalid exports, duplicate nets, engine-load failures, and catastrophic static failures.
37. Do not select a parent until every parallel candidate from the same parent has completed, failed, or been voided.
38. Preserve reproducibility evidence; never delete the only recorded copy of a result.

## Git

39. Work directly on `main`; never create branches.
40. Never pull on `pwa-llm`; integrate by fetch + cherry-pick.
41. Avoid fixup commits; amend when practical.
42. Never add AI or bot co-author trailers.
43. Stage only requested files and verify commit identity.
44. Include SPRT Elo in the commit subject whenever a result exists.
45. Never modify other repositories unless explicitly given permission
46. Before a `pwa-5090` winner becomes a parent, transfer the entire `runs/{run}/` directory, verify the checkpoint SHA-256, and confirm `continue_from` resolves to the optimizer checkpoint rather than the exported-net fallback.
47. After canonical promotion, update `candidate.net` and the Forge reference net; never point either at an active, rejected, or foreign net.
