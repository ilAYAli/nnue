#!/usr/bin/env python3
from __future__ import annotations

import json
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


    def test_iteration_commit_keeps_accepted_build_and_leaves_next_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            build = tmp / "build.json"
            arch = tmp / "architecture.json"
            build.write_text(
                json.dumps({
                    "run": "uho-native-1.0.34",
                    "lineage": "scratch-native",
                    "wdl": 0.75,
                    "hypothesis": "previous",
                    "data": {
                        "source_binpack": "data/nodes5000pv2_UHO.binpack",
                        "limit": 100000000,
                        "offset": 400000000,
                    },
                }, indent=2) + "\n",
                encoding="utf-8",
            )
            arch.write_text('{"input_buckets":8}\n', encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=tmp, check=True)
            subprocess.run(["git", "config", "user.name", "Petter Wahlman"], cwd=tmp, check=True)
            subprocess.run(["git", "config", "user.email", "petter@wahlman.no"], cwd=tmp, check=True)
            subprocess.run(["git", "add", "build.json", "architecture.json"], cwd=tmp, check=True)
            subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=tmp, check=True)

            build.write_text(
                json.dumps({
                    "run": "uho-native-1.0.35",
                    "lineage": "scratch-native",
                    "continue_from": "uho-native-1.0.33",
                    "wdl": 0.75,
                    "hypothesis": "Continue accepted uho-native-1.0.33 using the next UHO data window",
                    "data": {
                        "source_binpack": "data/nodes5000pv2_UHO.binpack",
                        "limit": 100000000,
                        "offset": 500000000,
                    },
                }, indent=2) + "\n",
                encoding="utf-8",
            )
            source = (REPO / "nnue-run").read_text(encoding="utf-8")
            harness = source.split('case "$cmd" in', 1)[0] + '''
BUILD=build.json
ARCH=architecture.json
bump_build_json "uho-native-1.0.35" "6.0" "0.49/2.20 (22%)" "forge command" "Crucible result"
'''
            harness_path = tmp / "harness.sh"
            harness_path.write_text(harness, encoding="utf-8")

            subprocess.run(["bash", str(harness_path)], cwd=tmp, check=True)

            committed = subprocess.run(
                ["git", "show", "HEAD:build.json"],
                cwd=tmp,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout
            committed_json = json.loads(committed)
            self.assertEqual("uho-native-1.0.35", committed_json["run"])
            self.assertEqual("uho-native-1.0.33", committed_json["continue_from"])
            self.assertEqual(500000000, committed_json["data"]["offset"])

            working_json = json.loads(build.read_text(encoding="utf-8"))
            self.assertEqual("uho-native-1.0.36", working_json["run"])
            self.assertEqual(600000000, working_json["data"]["offset"])
            self.assertNotIn("continue_from", working_json)

            diff = subprocess.run(
                ["git", "--no-pager", "diff", "--no-color", "HEAD", "--", "build.json"],
                cwd=tmp,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout
            self.assertIn('-  "run": "uho-native-1.0.35"', diff)
            self.assertIn('+  "run": "uho-native-1.0.36"', diff)
            self.assertIn('-  "continue_from": "uho-native-1.0.33"', diff)
            self.assertIn('-    "offset": 500000000', diff)
            self.assertIn('+    "offset": 600000000', diff)

    def test_failed_iteration_commit_is_marked_failed_and_leaves_next_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            build = tmp / "build.json"
            arch = tmp / "architecture.json"
            build.write_text(
                json.dumps({
                    "run": "uho-native-1.0.35",
                    "lineage": "scratch-native",
                    "wdl": 0.75,
                    "hypothesis": "previous",
                    "data": {
                        "source_binpack": "data/nodes5000pv2_UHO.binpack",
                        "limit": 100000000,
                        "offset": 500000000,
                    },
                }, indent=2) + "\n",
                encoding="utf-8",
            )
            arch.write_text('{"input_buckets":8}\n', encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=tmp, check=True)
            subprocess.run(["git", "config", "user.name", "Petter Wahlman"], cwd=tmp, check=True)
            subprocess.run(["git", "config", "user.email", "petter@wahlman.no"], cwd=tmp, check=True)
            subprocess.run(["git", "add", "build.json", "architecture.json"], cwd=tmp, check=True)
            subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=tmp, check=True)

            build.write_text(
                json.dumps({
                    "run": "uho-native-1.0.36",
                    "lineage": "scratch-native",
                    "wdl": 0.75,
                    "hypothesis": "Continue accepted uho-native-1.0.35 using the next UHO data window",
                    "data": {
                        "source_binpack": "data/nodes5000pv2_UHO.binpack",
                        "limit": 100000000,
                        "offset": 500000000,
                    },
                }, indent=2) + "\n",
                encoding="utf-8",
            )
            source = (REPO / "nnue-run").read_text(encoding="utf-8")
            harness = source.split('case "$cmd" in', 1)[0] + """
BUILD=build.json
ARCH=architecture.json
fail_build_json "uho-native-1.0.36" "-25.2" "-0.32/2.20 (-15%)" "forge command" "Crucible result"
"""
            harness_path = tmp / "harness.sh"
            harness_path.write_text(harness, encoding="utf-8")

            subprocess.run(["bash", str(harness_path)], cwd=tmp, check=True)

            subject = subprocess.run(
                ["git", "show", "--no-patch", "--format=%s", "HEAD"],
                cwd=tmp,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.strip()
            self.assertEqual(
                "fail: uho-native-1.0.36.nn: Elo -25.2,LLR -0.32/2.20 (-15%)",
                subject,
            )

            committed_json = json.loads(subprocess.run(
                ["git", "show", "HEAD:build.json"],
                cwd=tmp,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout)
            self.assertEqual("uho-native-1.0.36", committed_json["run"])
            self.assertEqual(500000000, committed_json["data"]["offset"])

            working_json = json.loads(build.read_text(encoding="utf-8"))
            self.assertEqual("uho-native-1.0.37", working_json["run"])
            self.assertEqual(600000000, working_json["data"]["offset"])

    def test_smoke_gate_rejects_bad_elo_or_negative_llr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            source = (REPO / "nnue-run").read_text(encoding="utf-8")
            harness = source.split('case "$cmd" in', 1)[0] + """
SMOKE_FAIL_ELO=-15.0
SMOKE_FAIL_LLR=-0.20
check_smoke() {
  if smoke_sprt_failed "$1" "$2"; then
    printf '%s=fail\n' "$3"
  else
    printf '%s=pass:%s\n' "$3" "$?"
  fi
}
check_smoke -25.2 -0.32/2.20 bad_elo
check_smoke -5.0 -0.20/2.20 negative_llr
check_smoke -5.0 -0.10/2.20 mild_negative
check_smoke 4.0 -0.22/2.20 positive_elo
"""
            harness_path = tmp / "smoke_gate.sh"
            harness_path.write_text(harness, encoding="utf-8")

            proc = subprocess.run(
                ["bash", str(harness_path)],
                cwd=tmp,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )

            self.assertIn("bad_elo=fail", proc.stdout)
            self.assertIn("negative_llr=fail", proc.stdout)
            self.assertIn("mild_negative=pass:1", proc.stdout)
            self.assertIn("positive_elo=pass:1", proc.stdout)

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
