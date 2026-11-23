import os
from pathlib import Path

import numpy as np
import pytest

from tools.lib import enyo_net_v1 as v1


@pytest.mark.parametrize("dtype", [np.int16, np.int32])
def test_leb128_round_trips_the_type_range(dtype):
    info = np.iinfo(dtype)
    values = np.array([0, 1, -1, 63, 64, -64, -65, 8191, 8192, -8192, -8193,
                       info.max, info.min + 1, info.min], dtype=np.int64)
    max_bytes = v1._leb_bytes(dtype)
    data = v1.leb128_encode(values, max_bytes)
    assert np.array_equal(v1.leb128_decode(data, len(values), max_bytes), values)


def test_leb128_is_minimal():
    assert v1.leb128_encode(np.array([63, 64, -64, -65]), 3) == bytes([0x3F, 0xC0, 0x00, 0x40, 0xBF, 0x7F])


def test_leb128_decode_is_strict():
    with pytest.raises(ValueError):
        v1.leb128_decode(bytes([0x80, 0x80, 0x80, 0x01]), 1, 3)
    with pytest.raises(ValueError):
        v1.leb128_decode(bytes([0x01, 0x81]), 2, 3)
    with pytest.raises(ValueError):
        v1.leb128_decode(bytes([0x01, 0x01]), 1, 3)


def test_stockfish_hashes_match_the_current_variant():
    network, ft, fc = v1._sf_hashes(v1.SUPPORTED_LAYOUT)
    assert ft ^ fc == network
    assert fc == 0x63337116


def test_unsupported_layout_is_rejected():
    layout = dict(v1.SUPPORTED_LAYOUT, input_set=2)
    with pytest.raises(ValueError, match="unsupported layout input_set=2 .*supported: input_set=0[|]1"):
        v1.check_layout(layout)
    v1.check_layout(dict(v1.SUPPORTED_LAYOUT, input_set=1))


def test_input_set_1_sizes_the_king_rows_and_has_no_nnue_form():
    layout = dict(v1.SUPPORTED_LAYOUT, input_set=1)
    sections = {name: shape for name, _, shape, _ in v1.feature_sections(layout)}
    assert sections["halfka_weights"] == (32 * 12 * 64, 1024)
    assert sections["halfka_psqt"] == (32 * 12 * 64, 8)
    with pytest.raises(ValueError, match="no Stockfish .nnue form"):
        v1.to_nnue(v1.NetV1(v1.LAYER_STACKS, 0, 0, {}, {}, layout), "")


def test_fc0_64_and_phase_buckets_are_layouts_without_nnue_form():
    for fc0_out, scheme in ((64, 0), (32, 1), (64, 1)):
        for input_set in (0, 1):
            layout = dict(v1.SUPPORTED_LAYOUT, input_set=input_set, fc0_out=fc0_out, bucket_scheme=scheme)
            v1.check_layout(layout)
            sections = {name: shape for name, _, shape in v1.stack_sections(layout)}
            assert sections["fc0_weights"] == (fc0_out, 1024)
            assert sections["fc1_weights"] == (32, 2 * fc0_out)
            assert sections["fc2_weights"] == (2 * fc0_out + 64,)
            with pytest.raises(ValueError, match="no Stockfish .nnue form"):
                v1.to_nnue(v1.NetV1(v1.LAYER_STACKS, 0, 0, {}, {}, layout), "")
    with pytest.raises(ValueError, match="fc0_out=48 .*supported: .*hidden=1024[|]1536 .*fc0_out=32[|]64 .*bucket_scheme=0[|]1"):
        v1.check_layout(dict(v1.SUPPORTED_LAYOUT, fc0_out=48))
    with pytest.raises(ValueError, match="bucket_scheme=2"):
        v1.check_layout(dict(v1.SUPPORTED_LAYOUT, bucket_scheme=2))


def test_hidden_1536_is_a_layout_without_nnue_form():
    for fc0_out in (32, 64):
        for input_set in (0, 1):
            layout = dict(v1.SUPPORTED_LAYOUT, input_set=input_set, hidden=1536, fc0_out=fc0_out)
            v1.check_layout(layout)
            sections = {name: shape for name, _, shape in v1.stack_sections(layout)}
            assert sections["fc0_weights"] == (fc0_out, 1536)
            with pytest.raises(ValueError, match="no Stockfish .nnue form"):
                v1.to_nnue(v1.NetV1(v1.LAYER_STACKS, 0, 0, {}, {}, layout), "")
    with pytest.raises(ValueError, match="hidden=2048 .*supported: .*hidden=1024[|]1536"):
        v1.check_layout(dict(v1.SUPPORTED_LAYOUT, hidden=2048))


@pytest.mark.skipif(not os.environ.get("ENYO_NNUE_TEST_SOURCE"),
                    reason="ENYO_NNUE_TEST_SOURCE (a layer-stacks .nnue) is not set")
def test_nnue_converts_losslessly_both_ways():
    data = Path(os.environ["ENYO_NNUE_TEST_SOURCE"]).read_bytes()
    net, description = v1.from_nnue(data, {"name": "test", "trained": "2026-01-01T00:00Z"})
    container = v1.container_bytes(net)
    back = v1.parse(container)
    assert back.architecture == v1.LAYER_STACKS
    assert back.metadata == {"name": "test", "trained": "2026-01-01T00:00Z"}
    assert v1.to_nnue(back, description) == data
