# NNUE Tools

These are lower-level phase tools. Use `../nnue` with `../build.json` for
the normal net build workflow.

## Phase Tools

```sh
tools/posgen/posgen.py --help
tools/score/score.py --help
tools/pack/pack.py --help
tools/bullet/train --help
tools/validate/validate.py --help
```

Phase meanings:

```text
posgen      create/select positions
score       attach cp/WDL/search targets
pack        convert scored JSONL to tensors
train       train/export .nn candidates
validate    static checks, failure suite, replay, SPRT
```
