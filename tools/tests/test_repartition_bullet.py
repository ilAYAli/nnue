from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from tools.bullet.repartition_bullet import RECORD_BYTES, input_files, repartition


class RepartitionBulletTests(unittest.TestCase):
    def test_preserves_stream_and_balances_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            payload = bytes(range(256)) * 3
            for index, records in enumerate((3, 2, 4)):
                start = sum((3, 2, 4)[:index]) * RECORD_BYTES
                (source / f"shard-{index}.bullet").write_bytes(payload[start:start + records * RECORD_BYTES])
            manifest = repartition(input_files(source), output, 4)
            actual = b"".join((output / item["path"]).read_bytes() for item in manifest["chunks"])
            self.assertEqual(payload[:9 * RECORD_BYTES], actual)
            self.assertEqual(hashlib.sha256(actual).hexdigest(), manifest["stream_sha256"])
            self.assertEqual([3, 2, 2, 2], [item["records"] for item in manifest["chunks"]])


if __name__ == "__main__":
    unittest.main()
