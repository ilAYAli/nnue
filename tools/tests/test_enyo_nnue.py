from pathlib import Path

import numpy as np

from tools.lib import enyo_nnue as nn2
from tools.lib.nnue_model import EnyoNNUE, export_model, load_model_from_nn


def _zero_net(input_buckets: int) -> nn2.Net:
    return nn2.Net(
        input_weights=np.zeros(
            (nn2.feature_count(input_buckets), nn2.N_HIDDEN), dtype=np.int16),
        input_biases=np.zeros(nn2.N_HIDDEN, dtype=np.int16),
        l1_weights=np.zeros((nn2.N_L2, nn2.N_L1), dtype=np.int8),
        l1_biases=np.zeros(nn2.N_L2, dtype=np.int32),
        l2_weights=np.zeros((nn2.N_L3, nn2.N_L2), dtype=np.float32),
        l2_biases=np.zeros(nn2.N_L3, dtype=np.float32),
        output_weights=np.zeros(nn2.N_L3, dtype=np.float32),
        output_bias=0.0,
        input_buckets=input_buckets,
    )


def test_network_size_supports_16_and_32_buckets() -> None:
    assert nn2.network_size(16) == nn2.NETWORK_SIZE
    assert nn2.network_size(32) > nn2.network_size(16)
    assert nn2.detect_input_buckets(nn2.network_size(16)) == 16
    assert nn2.detect_input_buckets(nn2.network_size(32)) == 32


def test_32_bucket_net_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "zero32.nn"
    nn2.write_net(_zero_net(32), path)

    loaded = nn2.load_net(path)

    assert path.stat().st_size == nn2.network_size(32)
    assert loaded.input_buckets == 32
    assert loaded.input_weights.shape == (nn2.feature_count(32), nn2.N_HIDDEN)


def test_pytorch_model_load_and_export_preserve_bucket_count(tmp_path: Path) -> None:
    source = tmp_path / "source32.nn"
    exported = tmp_path / "exported32.nn"
    nn2.write_net(_zero_net(32), source)

    model = load_model_from_nn(source)
    export_model(model, exported)

    assert isinstance(model, EnyoNNUE)
    assert model.input_buckets == 32
    assert nn2.load_net(exported).input_buckets == 32
