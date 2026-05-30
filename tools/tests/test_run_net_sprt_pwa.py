#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import textwrap
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "validate" / "run_net_sprt_pwa.sh"


def write_sparse(path: Path, size: int) -> None:
    with path.open("wb") as handle:
        handle.seek(size - 1)
        handle.write(b"\0")


def write_executable(path: Path, text: str) -> None:
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
    path.chmod(0o755)


class RunNetSprtPwaTests(unittest.TestCase):
    def make_tools(self, tmp: Path) -> tuple[Path, Path]:
        engine = tmp / "fake_enyo"
        sprt = tmp / "fake_sprt"
        write_executable(
            engine,
            r"""
            #!/usr/bin/env bash
            set -euo pipefail
            input=$(cat)
            net=$(printf '%s\n' "$input" | sed -n 's/^evalnet load //p' | head -1)
            size=$(python3 - "$net" <<'PY'
import os
import sys
print(os.path.getsize(sys.argv[1]))
PY
)
            if [ "${FAIL_32:-0}" = "1" ] && [ "$size" = "50368836" ]; then
              echo "network: '$net' is 50368836 bytes, expected 25203012"
              echo "evalnet load fail $net"
              exit 0
            fi
            echo "id name fake"
            echo "uciok"
            echo "evalnet load ok $net (search routed through network)"
            """,
        )
        write_executable(
            sprt,
            r"""
            #!/usr/bin/env bash
            set -euo pipefail
            touch "${SPRT_MARKER:?}"
            echo "fake sprt"
            """,
        )
        return engine, sprt

    def run_script(
        self,
        tmp: Path,
        *,
        engine: Path,
        sprt: Path,
        candidate: Path,
        reference: Path,
        fail_32: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update({
            "NET": str(candidate),
            "INIT": str(reference),
            "ENGINE": str(engine),
            "SPRT": str(sprt),
            "SPRT_MARKER": str(tmp / "sprt-called"),
            "BOOK": "",
            "LOG_DIR": str(tmp / "log"),
            "TAG": "test-sprt",
            "GAMES": "2",
        })
        if fail_32:
            env["FAIL_32"] = "1"
        return subprocess.run(
            ["bash", str(SCRIPT)],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    def test_refuses_engine_that_cannot_load_32_bucket_net(self) -> None:
        with self.subTest("32-bucket candidate rejected before sprt"):
            import tempfile

            with tempfile.TemporaryDirectory() as tmp_name:
                tmp = Path(tmp_name)
                candidate = tmp / "candidate32.nn"
                reference = tmp / "reference16.nn"
                write_sparse(candidate, 50368836)
                write_sparse(reference, 25203012)
                engine, sprt = self.make_tools(tmp)

                result = self.run_script(
                    tmp,
                    engine=engine,
                    sprt=sprt,
                    candidate=candidate,
                    reference=reference,
                    fail_32=True,
                )

                self.assertNotEqual(result.returncode, 0)
                self.assertIn("engine cannot load candidate 32-bucket net", result.stdout)
                self.assertFalse((tmp / "sprt-called").exists())

    def test_checks_candidate_and_reference_before_sprt(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            candidate = tmp / "candidate32.nn"
            reference = tmp / "reference16.nn"
            write_sparse(candidate, 50368836)
            write_sparse(reference, 25203012)
            engine, sprt = self.make_tools(tmp)

            result = self.run_script(
                tmp,
                engine=engine,
                sprt=sprt,
                candidate=candidate,
                reference=reference,
            )

            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("load ok candidate 32-bucket", result.stdout)
            self.assertIn("load ok reference 16-bucket", result.stdout)
            self.assertTrue((tmp / "sprt-called").exists())

    def test_script_does_not_call_notifai_directly(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("notifai.sh", text)


if __name__ == "__main__":
    unittest.main()
