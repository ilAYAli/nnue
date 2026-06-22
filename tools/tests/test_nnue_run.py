#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


class NnueRunTests(unittest.TestCase):
    def test_shell_syntax_is_valid(self) -> None:
        subprocess.run(["bash", "-n", str(REPO / "nnue-run")], check=True)

    def test_help_documents_thin_interface(self) -> None:
        proc = subprocess.run(
            [str(REPO / "nnue-run"), "help"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        text = proc.stdout
        self.assertIn("./nnue-run iterate", text)
        self.assertIn("./nnue-run plan", text)
        self.assertIn("./nnue-run train", text)
        self.assertIn("./nnue-run gates", text)
        self.assertIn("./nnue-run sprt", text)
        self.assertIn("./nnue-run status", text)
        self.assertNotIn("start", text)
        self.assertNotIn("doctor", text)

    def test_sprt_refuses_to_queue_when_crucible_is_busy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            home = tmp / "home"
            nets = home / "assets" / "nets"
            nets.mkdir(parents=True)
            (nets / "candidate.nn").write_bytes(b"candidate")
            (nets / "reference.nn").write_bytes(b"reference")
            build = tmp / "build.json"
            build.write_text(
                '{"run":"candidate","continue_from":"reference"}\n',
                encoding="utf-8",
            )
            fake_crucible = tmp / "crucible"
            fake_crucible.write_text(
                "#!/usr/bin/env bash\n"
                "if [[ \"$1\" == \"status\" && \"$2\" == \"--json\" ]]; then\n"
                "  printf '%s\\n' "
                "'{\"runs\":[{\"run\":\"busy-run\",\"state\":\"running\","
                "\"done\":1,\"tasks\":2,\"progress_fields\":[\"games=50/100\",\"elo=+1.0\"]}]}'\n"
                "  exit 0\n"
                "fi\n"
                "echo \"unexpected crucible call: $*\" >&2\n"
                "exit 2\n",
                encoding="utf-8",
            )
            fake_crucible.chmod(0o755)
            env = os.environ.copy()
            env.update({
                "BUILD": str(build),
                "CRUCIBLE": str(fake_crucible),
                "HOME": str(home),
                "NNUE_NTFY": "0",
            })

            proc = subprocess.run(
                [str(REPO / "nnue-run"), "sprt"],
                cwd=REPO,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertNotEqual(0, proc.returncode)
            self.assertIn(
                "refusing to start Crucible SPRT while Crucible is busy: "
                "busy-run running games=50/100 elo=+1.0",
                proc.stderr,
            )


if __name__ == "__main__":
    unittest.main()
