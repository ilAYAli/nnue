# NNUE Tools

These are lower-level phase tools. Use `../nnue-run` plus `../build.json` for
the normal net build workflow.

Python tools here may orchestrate, inspect, and run small validations. Large
row processing in the normal workflow must go through compiled tools.

## Phase Tools

```sh
tools/posgen/posgen.py --help
tools/score/score.py --help
tools/pack/pack.py --help
../nnue-train --help
tools/validate/validate.py --help
tools/pipeline/pipeline.py --help
```

Phase meanings:

```text
posgen      create/select positions
score       attach cp/WDL/search targets
pack        convert scored JSONL to tensors
train       train/export .nn candidates
validate    static checks, failure suite, replay, SPRT
pipeline    config launch/status backend
```

Generic run events are emitted by `tools/pipeline/pipeline.py` and appended to
`events.jsonl`. If configured, the same event JSON is passed to an external hook
on stdin and in `NNUE_RUN_EVENT_JSON`.
