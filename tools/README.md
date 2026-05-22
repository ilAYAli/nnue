# NNUE Internal Tools

These are implementation details for `../build.py`.

Do not treat the number of files in this directory as the public workflow. The
normal candidate interface is:

```sh
../build.py -c build.json
../build.py status
../build.py report
```

Run phase tools directly only when debugging a specific pipeline step or adding
a new backend. If a command becomes part of the normal workflow, expose it
through `build.py` and document it in the root `README.md`.

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
