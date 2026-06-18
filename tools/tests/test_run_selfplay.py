#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "tools" / "posgen" / "run_selfplay.sh"


class RunSelfplayTests(unittest.TestCase):
    def test_generates_explicit_enyo_config_and_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            fake_fastchess = tmp / "fastchess"
            fake_fastchess.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$@\" > \"$FAKE_FASTCHESS_ARGS\"\n",
                encoding="utf-8",
            )
            fake_fastchess.chmod(0o755)

            fake_engine = tmp / "enyo"
            fake_engine.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            fake_engine.chmod(0o755)

            book = tmp / "book.epd"
            book.write_text("8/8/8/8/8/8/8/8 w - - 0 1\n", encoding="utf-8")
            nnue = tmp / "model.nn"
            nnue.write_bytes(b"dummy")
            output = tmp / "selfplay.pgn"
            args_file = tmp / "fastchess.args"

            proc = subprocess.run(
                [
                    str(RUNNER),
                    "--runner", str(fake_fastchess),
                    "--engine", str(fake_engine),
                    "--book", str(book),
                    "--nnue-file", str(nnue),
                    "--output", str(output),
                    "--games", "1",
                    "--threads", "3",
                    "--engine-option", "Hash=256",
                ],
                env={"FAKE_FASTCHESS_ARGS": str(args_file)},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            config = output.with_name("selfplay.enyo-config.json")
            wrapper = output.with_name("selfplay.enyo-wrapper.sh")
            self.assertTrue(config.exists(), proc.stdout + proc.stderr)
            self.assertTrue(wrapper.exists(), proc.stdout + proc.stderr)

            payload = json.loads(config.read_text(encoding="utf-8"))
            self.assertEqual(
                {"use_lmr": True, "use_nnue": True, "use_syzygy": False},
                payload["toggles"],
            )
            self.assertEqual(str(nnue), payload["constants"]["nnue_file"])
            self.assertEqual(3, payload["constants"]["threads"])
            self.assertEqual(256, payload["constants"]["Hash"])

            fastchess_args = args_file.read_text(encoding="utf-8")
            self.assertIn(f"cmd={wrapper}", fastchess_args)
            self.assertIn(f'exec "{fake_engine}" --config "{config}" "$@"', wrapper.read_text(encoding="utf-8"))

    def test_generated_config_disables_nnue_without_nnue_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            fake_fastchess = tmp / "fastchess"
            fake_fastchess.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            fake_fastchess.chmod(0o755)

            fake_engine = tmp / "enyo"
            fake_engine.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            fake_engine.chmod(0o755)

            book = tmp / "book.epd"
            book.write_text("8/8/8/8/8/8/8/8 w - - 0 1\n", encoding="utf-8")
            output = tmp / "selfplay.pgn"

            subprocess.run(
                [
                    str(RUNNER),
                    "--runner", str(fake_fastchess),
                    "--engine", str(fake_engine),
                    "--book", str(book),
                    "--output", str(output),
                    "--games", "1",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            payload = json.loads(
                output.with_name("selfplay.enyo-config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                {"use_lmr": True, "use_nnue": False, "use_syzygy": False},
                payload["toggles"],
            )
            self.assertNotIn("nnue_file", payload["constants"])


if __name__ == "__main__":
    unittest.main()
