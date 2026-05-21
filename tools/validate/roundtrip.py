from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import enyo_nnue as nn2
from lib.nnue_model import export_model, load_model_from_nn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("net", help="Input Berserk-format .nn")
    args = ap.parse_args()

    src = Path(args.net)
    model = load_model_from_nn(src)
    with tempfile.NamedTemporaryFile(suffix=".nn", delete=False) as tmp:
        out = Path(tmp.name)

    try:
        export_model(model, out)
        a = src.read_bytes()
        b = out.read_bytes()
        if len(a) == nn2.LEGACY_NETWORK_SIZE:
            old_net = nn2.load_net(src)
            new_net = nn2.load_net(out)
            if not (
                (old_net.input_weights == new_net.input_weights).all()
                and (old_net.input_biases == new_net.input_biases).all()
                and (old_net.l1_weights == new_net.l1_weights).all()
                and (old_net.l1_biases == new_net.l1_biases).all()
                and (old_net.l2_weights == new_net.l2_weights).all()
                and (old_net.l2_biases == new_net.l2_biases).all()
                and (old_net.output_weights == new_net.output_weights).all()
                and old_net.output_bias == new_net.output_bias
            ):
                raise SystemExit("legacy conversion mismatch")
            print(f"roundtrip ok: {src} legacy->material-phase")
        elif a != b:
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    raise SystemExit(
                        f"roundtrip mismatch at byte {i}: {x} != {y}")
            raise SystemExit(f"roundtrip size mismatch {len(a)} != {len(b)}")
        else:
            print(f"roundtrip ok: {src}")
    finally:
        out.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
