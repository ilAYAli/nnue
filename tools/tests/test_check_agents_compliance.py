from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools/validate/check_agents_compliance.py"


class AgentsComplianceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def _write(self, build: dict, lineage_line: str) -> tuple[Path, Path]:
        build_path = self.tmp / "build.json"
        lineage_path = self.tmp / "LINEAGE.md"
        build_path.write_text(json.dumps(build))
        lineage_path.write_text(lineage_line)
        return build_path, lineage_path

    def run_check(self, build_path: Path, lineage_path: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--build", str(build_path),
             "--lineage", str(lineage_path)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_passes_a_well_formed_build(self) -> None:
        build_path, lineage_path = self._write(
            {"run": "enyo-16.0.0-rc1", "hypothesis": "test",
             "initialize_from": "enyo-7.4.0-rc1"},
            "Reserved: `enyo-16.0.0-rc1` on pwa-llm")
        proc = self.run_check(build_path, lineage_path)
        self.assertEqual(0, proc.returncode, proc.stdout + proc.stderr)

    def test_rejects_both_origin_fields(self) -> None:
        build_path, lineage_path = self._write(
            {"run": "enyo-16.0.0-rc1", "hypothesis": "test",
             "initialize_from": "enyo-7.4.0-rc1",
             "continue_from": "enyo-7.4.0-rc1"},
            "Reserved: `enyo-16.0.0-rc1` on pwa-llm")
        proc = self.run_check(build_path, lineage_path)
        self.assertNotEqual(0, proc.returncode)
        self.assertIn("both continue_from and initialize_from", proc.stdout)

    def test_rejects_bad_run_name_format(self) -> None:
        build_path, lineage_path = self._write(
            {"run": "enyo-fullthreats-coverage-check-rc1", "hypothesis": "test"},
            "Reserved: `enyo-fullthreats-coverage-check-rc1` on pwa-llm")
        proc = self.run_check(build_path, lineage_path)
        self.assertNotEqual(0, proc.returncode)
        self.assertIn("does not match", proc.stdout)

    def test_rejects_unreserved_run_name(self) -> None:
        build_path, lineage_path = self._write(
            {"run": "enyo-16.0.0-rc1", "hypothesis": "test"},
            "nothing reserved here")
        proc = self.run_check(build_path, lineage_path)
        self.assertNotEqual(0, proc.returncode)
        self.assertIn("is not mentioned in LINEAGE.md", proc.stdout)

    def test_rejects_missing_hypothesis(self) -> None:
        build_path, lineage_path = self._write(
            {"run": "enyo-16.0.0-rc1", "hypothesis": ""},
            "Reserved: `enyo-16.0.0-rc1` on pwa-llm")
        proc = self.run_check(build_path, lineage_path)
        self.assertNotEqual(0, proc.returncode)
        self.assertIn("missing a non-empty", proc.stdout)


if __name__ == "__main__":
    unittest.main()
