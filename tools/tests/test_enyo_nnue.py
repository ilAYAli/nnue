from pathlib import Path

import numpy as np
import torch

from tools.lib import enyo_nnue as nn2
from tools.lib.nnue_model import EnyoNNUE, export_model, load_model_from_nn


def _zero_net(
    input_buckets: int,
    feature_channels: int = nn2.DEFAULT_N_FEATURE_CHANNELS,
    output_buckets: int = 1,
    output_head_features: int = 0,
) -> nn2.Net:
    return nn2.Net(
        input_weights=np.zeros(
            (nn2.feature_count(input_buckets, feature_channels), nn2.N_HIDDEN),
            dtype=np.int16),
        input_biases=np.zeros(nn2.N_HIDDEN, dtype=np.int16),
        l1_weights=np.zeros((nn2.N_L2, nn2.N_L1), dtype=np.int8),
        l1_biases=np.zeros(nn2.N_L2, dtype=np.int32),
        l2_weights=np.zeros((nn2.N_L3, nn2.N_L2), dtype=np.float32),
        l2_biases=np.zeros(nn2.N_L3, dtype=np.float32),
        output_weights=np.zeros(
            (output_buckets, nn2.N_L3 + output_head_features),
            dtype=np.float32),
        output_biases=np.zeros(output_buckets, dtype=np.float32),
        input_buckets=input_buckets,
        feature_channels=feature_channels,
        output_buckets=output_buckets,
        output_head_features=output_head_features,
    )


def test_network_size_supports_16_and_32_buckets() -> None:
    assert nn2.network_size(16) == nn2.NETWORK_SIZE
    assert nn2.network_size(32) > nn2.network_size(16)
    assert nn2.detect_input_buckets(nn2.network_size(16)) == 16
    assert nn2.detect_input_buckets(nn2.network_size(32)) == 32
    assert nn2.detect_feature_channels(nn2.network_size(16)) == 12
    assert nn2.detect_feature_channels(nn2.network_size(32)) == 12
    assert nn2.detect_feature_channels(nn2.network_size(32, 1, 0, 11)) == 11
    assert nn2.detect_output_buckets(nn2.network_size(16, 4)) == 4
    assert nn2.detect_output_head_features(
        nn2.network_size(16, 4, nn2.N_HEAD_FEATURES)
    ) == nn2.N_HEAD_FEATURES


def test_32_bucket_net_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "zero32.nn"
    nn2.write_net(_zero_net(32), path)

    loaded = nn2.load_net(path)

    assert path.stat().st_size == nn2.network_size(32)
    assert loaded.input_buckets == 32
    assert loaded.feature_channels == 12
    assert loaded.input_weights.shape == (nn2.feature_count(32), nn2.N_HIDDEN)


def test_halfka_v2_channel_net_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "zero32x11.nn"
    nn2.write_net(_zero_net(32, feature_channels=11), path)

    loaded = nn2.load_net(path)

    assert path.stat().st_size == nn2.network_size(32, 1, 0, 11)
    assert loaded.input_buckets == 32
    assert loaded.feature_channels == 11
    assert loaded.input_weights.shape == (
        nn2.feature_count(32, 11), nn2.N_HIDDEN)


def test_halfka_v2_feature_channels_merge_kings_only() -> None:
    assert nn2.feature_count(32, 11) == 22528
    assert nn2.feature_channel(nn2.KING, nn2.WHITE, nn2.WHITE, 11) == 10
    assert nn2.feature_channel(nn2.KING, nn2.BLACK, nn2.WHITE, 11) == 10
    assert nn2.feature_channel(nn2.KING, nn2.WHITE, nn2.BLACK, 11) == 10
    assert nn2.feature_channel(nn2.KING, nn2.BLACK, nn2.BLACK, 11) == 10
    assert nn2.feature_channel(nn2.PAWN, nn2.WHITE, nn2.WHITE, 11) == 0
    assert nn2.feature_channel(nn2.QUEEN, nn2.WHITE, nn2.WHITE, 11) == 4
    assert nn2.feature_channel(nn2.PAWN, nn2.BLACK, nn2.WHITE, 11) == 5
    assert nn2.feature_channel(nn2.QUEEN, nn2.BLACK, nn2.WHITE, 11) == 9
    assert nn2.feature_channel(nn2.PAWN, nn2.BLACK, nn2.BLACK, 11) == 0
    assert nn2.feature_channel(nn2.QUEEN, nn2.WHITE, nn2.BLACK, 11) == 9

    assert nn2.feature_index(
        nn2.KING, nn2.WHITE, 0, 4, nn2.WHITE, 32, 12
    ) != nn2.feature_index(
        nn2.KING, nn2.BLACK, 0, 4, nn2.WHITE, 32, 12)
    assert nn2.feature_index(
        nn2.KING, nn2.WHITE, 0, 4, nn2.WHITE, 32, 11
    ) == nn2.feature_index(
        nn2.KING, nn2.BLACK, 0, 4, nn2.WHITE, 32, 11)


def test_feature_index_uses_horizontal_mirroring() -> None:
    for feature_channels in (12, 11):
        input_buckets = 32 if feature_channels == 11 else 16
        for piece_type in (
            nn2.PAWN, nn2.KNIGHT, nn2.BISHOP, nn2.ROOK, nn2.QUEEN, nn2.KING,
        ):
            for color in (nn2.WHITE, nn2.BLACK):
                for view in (nn2.WHITE, nn2.BLACK):
                    sq = 9
                    king_sq = 3
                    assert nn2.feature_index(
                        piece_type, color, sq, king_sq, view,
                        input_buckets, feature_channels
                    ) == nn2.feature_index(
                        piece_type, color, sq ^ 7, king_sq ^ 7, view,
                        input_buckets, feature_channels)


def test_pytorch_model_load_and_export_preserve_bucket_count(tmp_path: Path) -> None:
    source = tmp_path / "source32.nn"
    exported = tmp_path / "exported32.nn"
    nn2.write_net(_zero_net(32), source)

    model = load_model_from_nn(source)
    export_model(model, exported)

    assert isinstance(model, EnyoNNUE)
    assert model.input_buckets == 32
    assert model.feature_channels == 12
    assert nn2.load_net(exported).input_buckets == 32


def test_export_model_does_not_move_module_to_cpu(tmp_path: Path) -> None:
    source = tmp_path / "source.nn"
    exported = tmp_path / "exported.nn"
    nn2.write_net(_zero_net(16), source)
    model = load_model_from_nn(source)

    def fail_cpu():
        raise AssertionError("export_model should copy tensors, not move module")

    model.cpu = fail_cpu  # type: ignore[method-assign]
    export_model(model, exported)

    assert nn2.load_net(exported).input_buckets == 16


def test_pytorch_model_expands_legacy_net_to_zero_material_head(
        tmp_path: Path) -> None:
    source = tmp_path / "legacy.nn"
    exported = tmp_path / "head.nn"
    net = _zero_net(16)
    net.output_biases[:] = np.asarray([64.0], dtype=np.float32)
    nn2.write_net(net, source)

    legacy = load_model_from_nn(source)
    expanded = load_model_from_nn(
        source, output_head_features=nn2.N_HEAD_FEATURES)
    counts = [32]
    offsets = np.asarray([0])
    feats = np.zeros(sum(counts), dtype=np.int64)
    args = (
        torch.from_numpy(feats),
        torch.from_numpy(feats),
        torch.tensor(offsets, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
        torch.zeros(len(counts), dtype=torch.long),
        torch.ones(len(counts), dtype=torch.float32),
    )

    np.testing.assert_allclose(
        expanded(*args).detach().numpy(),
        legacy(*args).detach().numpy(),
    )
    export_model(expanded, exported)
    loaded = nn2.load_net(exported)
    assert loaded.output_head_features == nn2.N_HEAD_FEATURES
    np.testing.assert_array_equal(
        loaded.output_weights[:, nn2.N_L3:],
        np.zeros((1, nn2.N_HEAD_FEATURES), dtype=np.float32),
    )


def test_pytorch_model_applies_material_head_features(tmp_path: Path) -> None:
    source = tmp_path / "material-head.nn"
    net = _zero_net(16, output_head_features=nn2.N_HEAD_FEATURES)
    net.output_weights[0, nn2.N_L3:] = np.asarray([32.0, 64.0], dtype=np.float32)
    nn2.write_net(net, source)

    model = load_model_from_nn(source)
    counts = [32]
    offsets = np.asarray([0])
    feats = np.zeros(sum(counts), dtype=np.int64)
    pred = model(
        torch.from_numpy(feats),
        torch.from_numpy(feats),
        torch.tensor(offsets, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
        torch.zeros(len(counts), dtype=torch.long),
        torch.tensor([1.5], dtype=torch.float32),
        piece_count=torch.tensor(counts, dtype=torch.long),
    )

    # raw = (32 * 0.5 + 64 * 1.0) / 32, then existing phase scaling.
    np.testing.assert_allclose(
        pred.detach().numpy(),
        np.asarray([3.75], dtype=np.float32),
    )


def test_pytorch_model_selects_material_output_bucket(tmp_path: Path) -> None:
    source = tmp_path / "bucketed.nn"
    net = _zero_net(16, output_buckets=4)
    net.output_biases[:] = np.asarray([0.0, 32.0, 64.0, 96.0], dtype=np.float32)
    nn2.write_net(net, source)

    model = load_model_from_nn(source)
    counts = [2, 8, 16, 32]
    offsets = np.cumsum([0] + counts[:-1])
    feats = np.zeros(sum(counts), dtype=np.int64)
    pred = model(
        torch.from_numpy(feats),
        torch.from_numpy(feats),
        torch.tensor(offsets, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
        torch.zeros(len(counts), dtype=torch.long),
        torch.ones(len(counts), dtype=torch.float32),
    )

    np.testing.assert_allclose(
        pred.detach().numpy(),
        np.asarray([0.0, 0.0, 1.0, 3.0], dtype=np.float32),
    )
