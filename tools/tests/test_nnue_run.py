#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


class NnueRunTests(unittest.TestCase):
    def run_sourced(self, body: str) -> subprocess.CompletedProcess[str]:
        text = (REPO / "nnue-run").read_text(encoding="utf-8")
        prefix = text.split(chr(10) + 'case "$cmd" in' + chr(10), 1)[0]
        return subprocess.run(
            ["bash", "-s"],
            input=f"{prefix}\n{body}\n",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

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

    def test_train_helper_rejects_stale_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            helper = Path(tmp) / "train"
            helper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            helper.chmod(0o755)
            os.utime(helper, (1, 1))

            proc = self.run_sourced(
                f"NNUE_NTFY=0; TRAIN_HELPER={shlex.quote(str(helper))}; "
                "ensure_train_helper_current"
            )

        self.assertNotEqual(0, proc.returncode)
        self.assertIn("stale trainer helper", proc.stderr)
        self.assertIn("tools/bullet/spike_trainer/", proc.stderr)


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
            self.assertEqual("uho-native-1.0.35", working_json["continue_from"])
            self.assertEqual(600000000, working_json["data"]["offset"])

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
            self.assertIn('+  "continue_from": "uho-native-1.0.35"', diff)
            self.assertIn('-    "offset": 500000000', diff)
            self.assertIn('+    "offset": 600000000', diff)

    def test_failed_iteration_commit_keeps_previous_good_base(self) -> None:
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
                    "continue_from": "uho-native-1.0.35",
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
continue_from=uho-native-1.0.35
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
            self.assertEqual("uho-native-1.0.35", committed_json["continue_from"])
            self.assertEqual(500000000, committed_json["data"]["offset"])

            working_json = json.loads(build.read_text(encoding="utf-8"))
            self.assertEqual("uho-native-1.0.37", working_json["run"])
            self.assertEqual("uho-native-1.0.35", working_json["continue_from"])
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

    def test_sprt_waits_matching_active_crucible_run_for_build_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            home = tmp / "home"
            forge = home / "code" / "cpp" / "chess" / "crucible" / "scripts" / "forge"
            forge.parent.mkdir(parents=True)
            forge.write_text(
                "#!/usr/bin/env bash\n"
                "echo forge-should-not-run >&2\n"
                "exit 9\n",
                encoding="utf-8",
            )
            forge.chmod(0o755)
            (home / "assets" / "nets").mkdir(parents=True)
            (home / "assets" / "nets" / "candidate.nn").write_bytes(b"candidate")
            (home / "assets" / "nets" / "reference.nn").write_bytes(b"reference")

            (tmp / "runs" / "candidate" / "logs").mkdir(parents=True)
            build = tmp / "build.json"
            build.write_text(
                '{"run":"candidate","continue_from":"reference","hypothesis":"active existing"}\n',
                encoding="utf-8",
            )

            fake_crucible = tmp / "crucible"
            fake_crucible.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "if [[ \"$1\" == status && \"${2:-}\" == --json ]]; then\n"
                "  printf '%s\\n' '{\"runs\":[{\"run\":\"candidate-sprt-3000-20260623-123456\",\"state\":\"current\",\"done\":1,\"tasks\":2,\"manifest\":\"/tmp/existing-manifest.json\",\"progress_fields\":[\"games=1200/3000\",\"elo=+1.0\",\"llr=0.01/2.20 (0%)\"]}]}'\n"
                "  exit 0\n"
                "fi\n"
                "if [[ \"$1\" == status && \"$2\" == candidate-sprt-3000-20260623-123456 ]]; then\n"
                "  if [[ -f \"$FAKE_DONE\" ]]; then\n"
                "    printf '%s\\n' '{\"state\":\"done\",\"progress_fields\":[\"games=3000/3000\",\"elo=+4.2\",\"llr=0.31/2.20 (14%)\"]}'\n"
                "  else\n"
                "    printf '%s\\n' '{\"state\":\"current\",\"manifest\":\"/tmp/existing-manifest.json\",\"progress_fields\":[\"games=1200/3000\",\"elo=+1.0\",\"llr=0.01/2.20 (0%)\"]}'\n"
                "  fi\n"
                "  exit 0\n"
                "fi\n"
                "if [[ \"$1\" == wait ]]; then\n"
                "  printf '%s\\n' \"$*\" >> \"$FAKE_CALLS\"\n"
                "  touch \"$FAKE_DONE\"\n"
                "  exit 0\n"
                "fi\n"
                "echo unexpected crucible call: $* >&2\n"
                "exit 2\n",
                encoding="utf-8",
            )
            fake_crucible.chmod(0o755)
            calls = tmp / "calls.txt"
            done = tmp / "done"

            source = (REPO / "nnue-run").read_text(encoding="utf-8")
            harness = source.split('case "$cmd" in', 1)[0] + """
NNUE_NTFY=0
SOLO=0
BUILD="$TEST_BUILD"
CRUCIBLE="$TEST_CRUCIBLE"
HOME="$TEST_HOME"
ENGINE="$HOME/assets/engines/enyo_91ede5f"
run=candidate
continue_from=reference
reference_net="$HOME/assets/nets/reference.nn"
candidate_net="$HOME/assets/nets/candidate.nn"
log_dir="runs/candidate/logs"
run_sprt_once sprt 3000 sprt
printf 'elo=%s\n' "$last_sprt_elo"
printf 'llr=%s\n' "$last_sprt_llr"
"""
            harness_path = tmp / "harness.sh"
            harness_path.write_text(harness, encoding="utf-8")
            env = os.environ.copy()
            env.update({
                "TEST_BUILD": str(build),
                "TEST_CRUCIBLE": str(fake_crucible),
                "TEST_HOME": str(home),
                "FAKE_CALLS": str(calls),
                "FAKE_DONE": str(done),
            })

            proc = subprocess.run(
                ["bash", str(harness_path)],
                cwd=tmp,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            self.assertIn("elo=4.2", proc.stdout)
            self.assertIn("llr=0.31/2.20 (14%)", proc.stdout)
            self.assertNotIn("forge-should-not-run", proc.stderr)
            self.assertEqual(
                "wait --manifest /tmp/existing-manifest.json\n",
                calls.read_text(encoding="utf-8"),
            )

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
                "  exit 1\n"
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
