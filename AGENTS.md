# Agent Rules

This repository uses skills defined in ~/.agents/skills/.
Read the applicable skill before acting; when more than one applies, read all of them.

If I ask a question, answer it without making changes. You may perform read-only inspection
needed to answer accurately, but do not edit files, commit, start/stop jobs, or otherwise
change state unless I explicitly ask.


## Shell

All long-running commands MUST be run in the `nnue_cmd` tmux session on pwa-llm with
output must be visible to the user.
You can also use the `nnue_cmd` tmux session on pwa-5090 to run training in parallel with pwa-llm
to save time. This can e.g be to check viability of features, ...

`pwa-llm` is the sole Forge coordinator. Launch every Forge command there,
including SPRT, benchmarks, synchronization, deployment, status, and stop
operations. `pwa-5090` may train candidates and participate as a Forge worker,
but must never coordinate a Forge run.

## Run names

Format: `enyo-{architecture_number}.{promotion_number}.0-rc{iteration}`

1) `architecture_number`: increments when `architecture.json` changes; resets `promotion_number` to 0 and `iteration` to 1.
2) `promotion_number`: increments when a candidate is accepted.
3) `iteration`: increments for each new RC within a promotion.
4) Every iteration requires a matching `build.json` change with the corresponding run name.

## Experiment contract

5) One meaningful variable change per iteration. Never test multiple things at once.
6) `build.json` must contain a concise `hypothesis` that explains why, not how.
7) The only tracked diff during training should be `build.json`
   Exceptions to this rule are `architecture.json` (for architecture experiments), or
   other changes that are needed for the built/iteration to succeed (Bullet, ...)
8) All other changes should be commited individually to avoid breaking rule 7)
9) A scratch root records an absolute baseline against `nn-0ee0657fb25e.nnue`.
   Descendants require a 5,000-game parent-relative match; shorter matches only
   screen. A decisive direct win selects the candidate. If direct play is
   inconclusive, compare fixed-SF Elo under identical conditions and select the
   stronger net; retain the incumbent if neither is better. Every selected
   candidate records the same fixed-SF benchmark. Commit each result with the
   files from 7) and its parent-relative Elo, or the root's SF Elo, in the subject.
10) Export, distinct-net, engine-load, start-position, and catastrophic static
   checks are integrity gates. Residual improvement is report-only. Game results
   decide promotion.

## Events and launch

11) Never poll. Events arrive via `llmsh` with tag `ai-in` (`done`, `fail`). One status check allowed.
12) On each event: fix the smallest defect and relaunch, pick one new hypothesis after rejection, or advance data after acceptance.
- Launch exactly:
  ```sh
  cd ~/code/chess/nnue
  HOOK_EVENTS=done,fail MIN_SLOPE=0.05 SKIP_SMOKE=1 GAMES=5000 ./nnue iterate
  ```
13) When a promotion candidate has been selected, add it to LINEAGE.md

`ping` permits one status check on both hosts. Leave active work untouched; handle a completed
or failed run as its event. Never launch a duplicate.

## Parallel-host promotion and handoff

- `pwa-llm` owns canonical `main`, Forge, numbering, and promotion. Its scratch
  root is sole; pwa-5090 siblings use the same accepted parent.
- Reserve each sibling name and host in `LINEAGE.md`; names are never reused.
  Keep `AUTO_ADVANCE` off and configure the next run only after reconciliation.
- Commit a completed pwa-5090 configuration with its SPRT result and preserve its
  net, provenance, hashes, data identity, and result. Do not modify dirty pwa-llm;
  integrate remote commits there, in reserved order, at an event boundary. Never pull.
- Wait for all siblings. Rejections remain recorded and cannot become parents.
  A decisive 5,000-game winner advances; otherwise use identical fixed-SF results
  as the tie-break. If none beats the incumbent, use the next RC.
- Before using a pwa-5090 winner, transfer its net and provenance to pwa-llm,
  verify hashes, integrate it, and verify `continue_from` resolves.
- A resumable infrastructure failure keeps its RC only unchanged; a void run consumes
  it. After four valid failures from one parent, reserve the next architecture feature
  on pwa-5090 beside one final conventional pwa-llm candidate.
