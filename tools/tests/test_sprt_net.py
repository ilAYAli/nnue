from __future__ import annotations

import importlib.util
import json
import subprocess
import unittest
from pathlib import Path
from unittest import mock


REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "validate" / "sprt_net.py"
SPEC = importlib.util.spec_from_file_location("sprt_net", TOOL)
assert SPEC is not None and SPEC.loader is not None
sprt_net = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sprt_net)


class ExistingRunTests(unittest.TestCase):
    def status(self, *, reference: str = "~/assets/nets/reference.nn") -> subprocess.CompletedProcess[str]:
        payload = {
            "commands": [{
                "command": (
                    "forge sprt --candidate-net ~/assets/nets/candidate.nn "
                    f"--reference-net {reference} --games 5000"
                )
            }]
        }
        return subprocess.CompletedProcess(
            args=["forge", "status"],
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )

    def test_identical_existing_run_matches(self) -> None:
        with mock.patch.object(sprt_net.subprocess, "run", return_value=self.status()):
            matches = sprt_net.existing_run_matches(
                "sprt-test",
                candidate_net=Path("/home/petter/assets/nets/candidate.nn"),
                reference_net=Path("/home/petter/assets/nets/reference.nn"),
                games=5000,
            )

        self.assertTrue(matches)

    def test_existing_run_with_other_reference_is_rejected(self) -> None:
        with mock.patch.object(
            sprt_net.subprocess,
            "run",
            return_value=self.status(reference="~/assets/nets/other.nn"),
        ):
            matches = sprt_net.existing_run_matches(
                "sprt-test",
                candidate_net=Path("/home/petter/assets/nets/candidate.nn"),
                reference_net=Path("/home/petter/assets/nets/reference.nn"),
                games=5000,
            )

        self.assertFalse(matches)


if __name__ == "__main__":
    unittest.main()
