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


class RunSprtTests(unittest.TestCase):
    def test_waits_and_prints_completed_progress(self) -> None:
        launch = mock.Mock()
        launch.stdout = iter(["run: id=sprt-test\n"])
        launch.wait.return_value = 0
        completed = subprocess.CompletedProcess(
            args=["forge", "status"],
            returncode=0,
            stdout=json.dumps({"progress": "progress=games=4000/4000, elo=-165.1, ci=7.4"}),
            stderr="",
        )
        with (
            mock.patch.object(sprt_net.subprocess, "Popen", return_value=launch),
            mock.patch.object(
                sprt_net.subprocess,
                "run",
                side_effect=[subprocess.CompletedProcess(args=["forge", "wait"], returncode=0), completed],
            ) as run,
            mock.patch("builtins.print") as printed,
        ):
            result = sprt_net.run_sprt(
                engine=Path("/engine"),
                candidate_net=Path("/candidate.nn"),
                reference_net=Path("/reference.nn"),
                games=4000,
            )

        self.assertEqual("sprt-test", result)
        self.assertEqual(
            [mock.call(["forge", "wait", "--manifest",
                       str(Path.home() / "code" / "chess" / "forge" / "runs" / "sprt-test" / "manifest.json")]),
             mock.call(["forge", "status", "sprt-test", "--json"], capture_output=True, text=True, check=True)],
            run.call_args_list,
        )
        printed.assert_any_call("progress=games=4000/4000, elo=-165.1, ci=7.4")


if __name__ == "__main__":
    unittest.main()
