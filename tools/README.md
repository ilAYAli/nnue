# NNUE Tools

These are lower-level phase tools. Use `../build.py` for the normal net build
workflow.

## Phase Tools

```sh
tools/posgen/posgen.py --help
tools/score/score.py --help
tools/pack/pack.py --help
tools/train/train.py --help
tools/validate/validate.py --help
tools/pipeline/pipeline.py --help
tools/bullet/bullet.py --help
```

Phase meanings:

```text
posgen      create/select positions
score       attach cp/WDL/search targets
pack        convert scored JSONL to tensors
train       train/export .nn candidates
validate    static checks, failure suite, replay, SPRT
pipeline    config launch/status backend
bullet      experimental BulletFormat conversion/training backend
```

Generic run events are emitted by `tools/pipeline/pipeline.py` and appended to
`events.jsonl`. If configured, the same event JSON is passed to an external hook
on stdin and in `NNUE_RUN_EVENT_JSON`.
