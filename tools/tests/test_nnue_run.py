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
        text = (REPO / "nnue").read_text(encoding="utf-8")
        prefix = text.split(chr(10) + 'case "$cmd" in' + chr(10), 1)[0]
        return subprocess.run(
            ["bash", "-s"],
            input=f"{prefix}\n{body}\n",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_shell_syntax_is_valid(self) -> None:
        subprocess.run(["bash", "-n", str(REPO / "nnue")], check=True)

    def test_help_documents_thin_interface(self) -> None:
        proc = subprocess.run(
            [str(REPO / "nnue"), "help"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        text = proc.stdout
        self.assertIn("./nnue iterate", text)
        self.assertIn("./nnue plan", text)
        self.assertIn("./nnue train", text)
        self.assertIn("./nnue gates", text)
        self.assertIn("./nnue sprt", text)
        self.assertIn("./nnue status", text)
        self.assertIn("SKIP_SMOKE=1 GAMES=800 ./nnue iterate", text)
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

    def test_distinct_net_gate_rejects_byte_identical_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            candidate = tmp / "candidate.nn"
            reference = tmp / "reference.nn"
            candidate.write_bytes(b"identical exported net")
            reference.write_bytes(b"identical exported net")

            proc = self.run_sourced(
                "NNUE_NTFY=0; "
                f"require_distinct_nets {shlex.quote(str(candidate))} "
                f"{shlex.quote(str(reference))}"
            )

        self.assertNotEqual(0, proc.returncode)
        self.assertIn("byte-identical to reference", proc.stderr)

    def test_distinct_net_gate_accepts_changed_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            candidate = tmp / "candidate.nn"
            reference = tmp / "reference.nn"
            candidate.write_bytes(b"candidate")
            reference.write_bytes(b"reference")

            proc = self.run_sourced(
                "NNUE_NTFY=0; "
                f"require_distinct_nets {shlex.quote(str(candidate))} "
                f"{shlex.quote(str(reference))}"
            )

        self.assertEqual("", proc.stderr)
        self.assertEqual(0, proc.returncode)

    def test_full_sprt_accepts_positive_at_cap(self) -> None:
        proc = self.run_sourced(
            "NNUE_NTFY=0; GAMES=3000; "
            "last_sprt_elo=16.0; last_sprt_llr='1.40/2.20 (64%)'; "
            "full_sprt_pass"
        )
        self.assertEqual("", proc.stderr)
        self.assertEqual(0, proc.returncode)

    def test_full_sprt_accepts_h1(self) -> None:
        proc = self.run_sourced(
            "NNUE_NTFY=0; GAMES=3000; "
            "last_sprt_elo=16.0; last_sprt_llr='2.20/2.20 (100%)'; "
            "full_sprt_pass"
        )
        self.assertEqual("", proc.stderr)
        self.assertEqual(0, proc.returncode)

    def test_full_sprt_rejects_negative_at_cap(self) -> None:
        proc = self.run_sourced(
            "NNUE_NTFY=0; GAMES=3000; "
            "last_sprt_elo=-1.0; last_sprt_llr='-0.01/2.20 (0%)'; "
            "full_sprt_pass"
        )
        self.assertNotEqual(0, proc.returncode)

    def test_net_for_name_or_path_expands_literal_home_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            home = Path(tmp_name) / "home"
            home.mkdir()
            proc = self.run_sourced(
                f"HOME={shlex.quote(str(home))}; "
                "net_for_name_or_path '~/assets/nets/nn-0ee0657fb25e.nnue'"
            )

            self.assertEqual("", proc.stderr)
            self.assertEqual(0, proc.returncode)
            self.assertEqual(
                f"{home}/assets/nets/nn-0ee0657fb25e.nnue\n",
                proc.stdout,
            )

    def test_load_config_defaults_reference_to_continue_from(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            home = tmp / "home"
            nets = home / "assets" / "nets"
            nets.mkdir(parents=True)
            (nets / "uho-native-1.0.42.nn").write_bytes(b"reference")
            build = tmp / "build.json"
            build.write_text(
                '{"run":"uho-native-1.0.43","continue_from":"uho-native-1.0.42"}\n',
                encoding="utf-8",
            )

            source = (REPO / "nnue").read_text(encoding="utf-8")
            harness = source.split('case "$cmd" in', 1)[0] + """
NNUE_NTFY=0
load_config
printf 'label=%s\n' "$(reference_label)"
printf 'net=%s\n' "$reference_net"
"""
            harness_path = tmp / "harness.sh"
            harness_path.write_text(harness, encoding="utf-8")
            env = os.environ.copy()
            env.update({
                "BUILD": str(build),
                "HOME": str(home),
                "NNUE_NTFY": "0",
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

            self.assertEqual("", proc.stderr)
            self.assertEqual(
                f"label=uho-native-1.0.42\nnet={nets}/uho-native-1.0.42.nn\n",
                proc.stdout,
            )

    def test_load_config_uses_shared_bullet_source_as_gate_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            build = tmp / "build.json"
            build.write_text(
                json.dumps({
                    "run": "candidate",
                    "data": {"source_binpack": "data/bullet/shared.bullet"},
                }),
                encoding="utf-8",
            )

            source = (REPO / "nnue").read_text(encoding="utf-8")
            harness = source.split('case "$cmd" in', 1)[0] + """
NNUE_NTFY=0
load_config
printf 'data=%s\n' "$data_file"
"""
            harness_path = tmp / "harness.sh"
            harness_path.write_text(harness, encoding="utf-8")
            env = os.environ.copy()
            env.update({"BUILD": str(build), "NNUE_NTFY": "0"})

            proc = subprocess.run(
                ["bash", str(harness_path)],
                cwd=tmp,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            self.assertEqual("", proc.stderr)
            self.assertEqual("data=data/bullet/shared.bullet\n", proc.stdout)

            build.write_text(
                json.dumps({
                    "run": "candidate",
                    "data": {
                        "source_binpack": "data/bullet/shared.bullet",
                        "offset": 1,
                        "limit": 2,
                    },
                }),
                encoding="utf-8",
            )
            proc = subprocess.run(
                ["bash", str(harness_path)],
                cwd=tmp,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            self.assertEqual("", proc.stderr)
            self.assertEqual("data=data/bullet/candidate.bullet\n", proc.stdout)

    def test_reference_net_env_overrides_continue_from(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            home = tmp / "home"
            nets = home / "assets" / "nets"
            nets.mkdir(parents=True)
            (nets / "uho-native-1.0.42.nn").write_bytes(b"parent")
            (nets / "nn-0ee0657fb25e.nnue").write_bytes(b"stockfish")
            build = tmp / "build.json"
            build.write_text(
                '{"run":"uho-native-1.0.43","continue_from":"uho-native-1.0.42"}\n',
                encoding="utf-8",
            )

            source = (REPO / "nnue").read_text(encoding="utf-8")
            harness = source.split('case "$cmd" in', 1)[0] + """
NNUE_NTFY=0
load_config
printf 'label=%s\n' "$(reference_label)"
printf 'net=%s\n' "$reference_net"
"""
            harness_path = tmp / "harness.sh"
            harness_path.write_text(harness, encoding="utf-8")
            env = os.environ.copy()
            env.update({
                "BUILD": str(build),
                "HOME": str(home),
                "NNUE_NTFY": "0",
                "REFERENCE_NET": "~/assets/nets/nn-0ee0657fb25e.nnue",
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

            self.assertEqual("", proc.stderr)
            self.assertEqual(
                f"label=~/assets/nets/nn-0ee0657fb25e.nnue\nnet={nets}/nn-0ee0657fb25e.nnue\n",
                proc.stdout,
            )

    def test_load_config_does_not_infer_previous_release_candidate_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            home = tmp / "home"
            nets = home / "assets" / "nets"
            nets.mkdir(parents=True)
            (nets / "native-2.0.0-rc13.nn").write_bytes(b"reference")
            build = tmp / "build.json"
            build.write_text('{"run":"native-2.0.0-rc14"}\n', encoding="utf-8")

            source = (REPO / "nnue").read_text(encoding="utf-8")
            harness = source.split('case "$cmd" in', 1)[0] + """
NNUE_NTFY=0
load_config
printf 'label=%s\n' "$(reference_label)"
printf 'net=%s\n' "$reference_net"
"""
            harness_path = tmp / "harness.sh"
            harness_path.write_text(harness, encoding="utf-8")
            env = os.environ.copy()
            env.update({
                "BUILD": str(build),
                "HOME": str(home),
                "NNUE_NTFY": "0",
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

            self.assertEqual("", proc.stderr)
            self.assertEqual(
                "label=\nnet=\n",
                proc.stdout,
            )

    def test_load_config_rejects_old_init_net_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            build = tmp / "build.json"
            build.write_text(
                '{"run":"candidate","init_net":"~/assets/nets/base.nn"}\n',
                encoding="utf-8",
            )

            source = (REPO / "nnue").read_text(encoding="utf-8")
            harness = source.split('case "$cmd" in', 1)[0] + """
NNUE_NTFY=0
load_config
"""
            harness_path = tmp / "harness.sh"
            harness_path.write_text(harness, encoding="utf-8")

            proc = subprocess.run(
                ["bash", str(harness_path)],
                cwd=tmp,
                env={**os.environ, "BUILD": str(build), "NNUE_NTFY": "0"},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertNotEqual(0, proc.returncode)
            self.assertIn("init_net was renamed to initialize_from", proc.stderr)

    def test_training_build_keeps_continue_from_with_initialize_from(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            home = tmp / "home"
            home.mkdir()
            build = tmp / "build.json"
            build.write_text(
                json.dumps({
                    "run": "uho-native-1.1.0",
                    "lineage": "native",
                    "continue_from": "uho-native-1.0.42",
                    "initialize_from": "~/assets/nets/uho-native-1.0.42.nn",
                    "reference": "uho-native-1.0.42",
                    "changed_variables": {"reference": "old"},
                    "data": {
                        "source_binpack": "data/nodes5000pv2_UHO.binpack",
                        "limit": 100000000,
                        "offset": 1200000000,
                    },
                }, indent=2) + "\n",
                encoding="utf-8",
            )

            source = (REPO / "nnue").read_text(encoding="utf-8")
            harness = source.split('case "$cmd" in', 1)[0] + """
NNUE_NTFY=0
load_config
training_build >/dev/null
"""
            harness_path = tmp / "harness.sh"
            harness_path.write_text(harness, encoding="utf-8")
            env = os.environ.copy()
            env.update({
                "BUILD": str(build),
                "HOME": str(home),
                "NNUE_NTFY": "0",
            })

            subprocess.run(
                ["bash", str(harness_path)],
                cwd=tmp,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            resolved = json.loads(
                (tmp / "runs" / "uho-native-1.1.0" / "build.resolved.json").read_text(encoding="utf-8")
            )
            self.assertEqual("uho-native-1.0.42", resolved["continue_from"])
            self.assertEqual("~/assets/nets/uho-native-1.0.42.nn", resolved["initialize_from"])
            self.assertNotIn("reference", resolved)
            self.assertNotIn("reference", resolved["changed_variables"])

    def test_training_build_rejects_missing_continue_from_non_interactive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            home = tmp / "home"
            home.mkdir()
            build = tmp / "build.json"
            build.write_text(
                json.dumps({
                    "run": "native-2.0.0-rc1",
                    "lineage": "native",
                    "hypothesis": "fresh scratch candidate",
                    "data": {
                        "source_binpack": "data/stockfish/master-binpacks/farseerT76.binpack",
                        "limit": 100000000,
                        "offset": 0,
                    },
                }, indent=2) + "\n",
                encoding="utf-8",
            )

            source = (REPO / "nnue").read_text(encoding="utf-8")
            harness = source.split('case "$cmd" in', 1)[0] + """
NNUE_NTFY=0
load_config
training_build >/dev/null
"""
            harness_path = tmp / "harness.sh"
            harness_path.write_text(harness, encoding="utf-8")
            env = os.environ.copy()
            env.update({
                "BUILD": str(build),
                "HOME": str(home),
                "NNUE_NTFY": "0",
            })

            proc = subprocess.run(
                ["bash", str(harness_path)],
                cwd=tmp,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertNotEqual(0, proc.returncode)
            self.assertIn("continue_from not found", proc.stderr)

    def test_train_propagates_helper_artifact_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            home = tmp / "home"
            nets = home / "assets" / "nets"
            nets.mkdir(parents=True)
            (nets / "candidate.nn").write_bytes(b"net")
            previous = tmp / "runs" / "base" / "checkpoints" / "native-1" / "optimiser_state"
            previous.mkdir(parents=True)
            (previous / "weights.bin").write_bytes(b"weights")
            build = tmp / "build.json"
            build.write_text('{"run":"candidate","continue_from":"base"}\n', encoding="utf-8")
            calls = tmp / "calls.txt"
            helper = tmp / "train-helper"
            helper.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$*\" >> \"$CALLS\"\n"
                "if [[ \"$1\" == plan ]]; then\n"
                "  printf '%s\\n' 'init_weights=runs/base/checkpoints/native-1/optimiser_state/weights.bin'\n"
                "  exit 0\n"
                "fi\n"
                "printf '%s\\n' 'error: stale train/export artifacts for candidate' >&2\n"
                "exit 9\n",
                encoding="utf-8",
            )
            helper.chmod(0o755)

            source = (REPO / "nnue").read_text(encoding="utf-8")
            harness = source.split('case "$cmd" in', 1)[0] + """
BUILD="$TEST_BUILD"
HOME="$TEST_HOME"
TRAIN_HELPER="$TEST_HELPER"
NNUE_NTFY=0
train
"""
            harness_path = tmp / "harness.sh"
            harness_path.write_text(harness, encoding="utf-8")
            env = os.environ.copy()
            env.update({
                "CALLS": str(calls),
                "TEST_BUILD": str(build),
                "TEST_HELPER": str(helper),
                "TEST_HOME": str(home),
            })

            proc = subprocess.run(
                ["bash", str(harness_path)],
                cwd=tmp,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertNotEqual(0, proc.returncode)
            self.assertIn(
                "stale train/export artifacts for candidate",
                proc.stdout + proc.stderr,
            )

    def test_train_propagates_helper_init_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            home = tmp / "home"
            (home / "assets" / "nets").mkdir(parents=True)
            previous = tmp / "runs" / "base" / "checkpoints" / "native-1" / "optimiser_state"
            previous.mkdir(parents=True)
            expected = previous / "weights.bin"
            expected.write_bytes(b"weights")
            build = tmp / "build.json"
            build.write_text('{"run":"candidate","continue_from":"base"}\n', encoding="utf-8")
            helper = tmp / "train-helper"
            helper.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "if [[ \"$1\" == plan ]]; then\n"
                f"  printf '%s\\n' 'init_weights={expected}'\n"
                "  exit 0\n"
                "fi\n"
                "if [[ \"$1\" == all ]]; then\n"
                f"  printf '%s\\n' 'error: training did not load expected init weights: {expected}' >&2\n"
                "  exit 1\n"
                "fi\n"
                "exit 9\n",
                encoding="utf-8",
            )
            helper.chmod(0o755)

            source = (REPO / "nnue").read_text(encoding="utf-8")
            harness = source.split('case "$cmd" in', 1)[0] + """
BUILD="$TEST_BUILD"
HOME="$TEST_HOME"
TRAIN_HELPER="$TEST_HELPER"
NNUE_NTFY=0
train
"""
            harness_path = tmp / "harness.sh"
            harness_path.write_text(harness, encoding="utf-8")
            env = os.environ.copy()
            env.update({
                "TEST_BUILD": str(build),
                "TEST_HELPER": str(helper),
                "TEST_HOME": str(home),
            })

            proc = subprocess.run(
                ["bash", str(harness_path)],
                cwd=tmp,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertNotEqual(0, proc.returncode)
            self.assertIn(
                f"training did not load expected init weights: {expected}",
                proc.stdout + proc.stderr,
            )

    def test_train_delegates_resume_to_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            home = tmp / "home"
            (home / "assets" / "nets").mkdir(parents=True)
            previous = tmp / "runs" / "base" / "checkpoints" / "native-1" / "optimiser_state"
            previous.mkdir(parents=True)
            expected = previous / "weights.bin"
            expected.write_bytes(b"weights")
            build = tmp / "build.json"
            build.write_text('{"run":"candidate","continue_from":"base"}\n', encoding="utf-8")
            calls = tmp / "calls.txt"
            helper = tmp / "train-helper"
            helper.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "printf '%s\\n' \"$1\" >> \"$CALLS\"\n"
                "if [[ \"$1\" == plan ]]; then\n"
                f"  printf '%s\\n' 'init_weights={expected}'\n"
                "  exit 0\n"
                "fi\n"
                "if [[ \"$1\" == all ]]; then\n"
                "  mkdir -p runs/candidate \"$HOME/assets/nets\"\n"
                "  printf model > runs/candidate/model.nn\n"
                "  printf candidate > \"$HOME/assets/nets/candidate.nn\"\n"
                f"  printf '%s\\n' 'loaded_init_weights={expected}'\n"
                "  exit 0\n"
                "fi\n"
                "if [[ \"$1\" == data ]]; then\n"
                "  mkdir -p data/bullet\n"
                "  printf data > data/bullet/candidate.bullet\n"
                "  exit 0\n"
                "fi\n"
                "exit 9\n",
                encoding="utf-8",
            )
            helper.chmod(0o755)

            source = (REPO / "nnue").read_text(encoding="utf-8")
            harness = source.split('case "$cmd" in', 1)[0] + """
BUILD="$TEST_BUILD"
HOME="$TEST_HOME"
TRAIN_HELPER="$TEST_HELPER"
NNUE_NTFY=0
train
train
"""
            harness_path = tmp / "harness.sh"
            harness_path.write_text(harness, encoding="utf-8")
            env = os.environ.copy()
            env.update({
                "CALLS": str(calls),
                "TEST_BUILD": str(build),
                "TEST_HELPER": str(helper),
                "TEST_HOME": str(home),
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

            self.assertFalse((tmp / "runs" / "candidate" / "train.provenance.json").exists())
            self.assertEqual("plan\nall\nplan\nall\n", calls.read_text(encoding="utf-8"))

    def test_next_run_name_bumps_release_candidates(self) -> None:
        proc = self.run_sourced('NNUE_NTFY=0; next_run_name "native-2.0.0-rc1"')
        self.assertEqual("", proc.stderr)
        self.assertEqual(0, proc.returncode)
        self.assertEqual("native-2.0.0-rc2\n", proc.stdout)

    def test_next_promoted_run_name_advances_minor_for_release_candidate(self) -> None:
        proc = self.run_sourced('NNUE_NTFY=0; next_promoted_run_name "native-3.0.0-rc7"')
        self.assertEqual("", proc.stderr)
        self.assertEqual(0, proc.returncode)
        self.assertEqual("native-3.1.0-rc1\n", proc.stdout)

    def test_stockfish_net_gate_reuses_champion_across_engine_versions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            ledger = tmp / "stockfish-net.jsonl"
            trace = tmp / "trace"
            ledger.write_text(
                '{"candidate":"champion","engine":"old-engine","reference":"nn-stockfish.nnue",'
                '"requested_games":500,"games":500,"elo":-158.9,"ci":28.4,'
                '"llr":-7.84,"llr_bound":690.78,"los":0.0,"draw":34.4}\n',
                encoding="utf-8",
            )

            proc = self.run_sourced(
                f"""
NNUE_NTFY=0
STOCKFISH_NET_LEDGER={shlex.quote(str(ledger))}
STOCKFISH_NET=nn-stockfish.nnue
STOCKFISH_NET_GAMES=500
PROMOTED_NET_LINK={shlex.quote(str(tmp / "candidate.net"))}
QSPRT=qsprt_mock
TRACE={shlex.quote(str(trace))}
ln -s champion.nn "$PROMOTED_NET_LINK"
qsprt_mock() {{
  printf '%s %s\\n' "$GAMES" "$*" >> "$TRACE"
  printf '{{"candidate":"challenger","engine":"engine","reference":"nn-stockfish.nnue","requested_games":500,"games":500,"elo":-150.0,"ci":28.0,"llr":-8.0,"llr_bound":690.78,"los":0.0,"draw":35.0}}\\n' >> "$STOCKFISH_NET_LEDGER"
}}
stockfish_net_gate challenger
"""
            )

            self.assertEqual("", proc.stderr)
            self.assertEqual(0, proc.returncode)
            self.assertEqual(
                "500 --candidate challenger.nn\n",
                trace.read_text(encoding="utf-8"),
            )
            self.assertIn("delta=8.9, upper90=35.0", proc.stdout)
            self.assertEqual(
                ["champion"],
                [json.loads(line)["candidate"] for line in ledger.read_text(encoding="utf-8").splitlines()],
            )

    def test_latest_stockfish_result_ignores_invalid_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            ledger = Path(tmp_name) / "stockfish-net.jsonl"
            ledger.write_text(
                '{"candidate":"candidate","reference":"nn-stockfish.nnue",'
                '"elo":-150.0,"valid":true}\n'
                '{"candidate":"candidate","reference":"nn-stockfish.nnue",'
                '"elo":-120.0,"valid":false}\n',
                encoding="utf-8",
            )

            proc = self.run_sourced(
                "NNUE_NTFY=0; "
                f"STOCKFISH_NET_LEDGER={shlex.quote(str(ledger))}; "
                "STOCKFISH_NET=nn-stockfish.nnue; "
                "latest_stockfish_result candidate"
            )

        self.assertEqual("", proc.stderr)
        self.assertEqual(0, proc.returncode)
        self.assertEqual(-150.0, json.loads(proc.stdout)["elo"])

    def test_stockfish_net_gate_vetoes_small_negative_delta_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            ledger = tmp / "stockfish-net.jsonl"
            ledger.write_text(
                '{"candidate":"champion","engine":"engine","reference":"nn-stockfish.nnue",'
                '"requested_games":500,"games":500,"elo":-174.5,"ci":28.2,'
                '"llr":-8.47,"llr_bound":690.78,"los":0.0,"draw":37.1}\n',
                encoding="utf-8",
            )

            proc = self.run_sourced(
                f"""
NNUE_NTFY=0
STOCKFISH_NET_LEDGER={shlex.quote(str(ledger))}
STOCKFISH_NET=nn-stockfish.nnue
STOCKFISH_NET_GAMES=500
PROMOTED_NET_LINK={shlex.quote(str(tmp / "candidate.net"))}
QSPRT=qsprt_mock
ln -s champion.nn "$PROMOTED_NET_LINK"
qsprt_mock() {{
  printf '{{"candidate":"challenger","engine":"engine","reference":"nn-stockfish.nnue","requested_games":500,"games":500,"elo":-180.8,"ci":28.5,"llr":-8.76,"llr_bound":690.78,"los":0.0,"draw":35.4}}\n' >> "$STOCKFISH_NET_LEDGER"
}}
if stockfish_net_gate challenger; then
  exit 9
fi
printf '%s %s\n' "$stockfish_delta" "$stockfish_upper90"
[[ $(readlink "$PROMOTED_NET_LINK") == champion.nn ]]
"""
            )

            self.assertEqual("", proc.stderr)
            self.assertEqual(0, proc.returncode)
            self.assertIn("-6.3 19.9", proc.stdout)
            self.assertEqual("champion.nn", os.readlink(tmp / "candidate.net"))

    def test_stockfish_net_gate_allows_configured_negative_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            ledger = tmp / "stockfish-net.jsonl"
            trace = tmp / "trace"
            ledger.write_text(
                '{"candidate":"champion","engine":"engine","reference":"nn-stockfish.nnue",'
                '"requested_games":500,"games":500,"elo":-174.5,"ci":28.2,'
                '"llr":-8.47,"llr_bound":690.78,"los":0.0,"draw":37.1}\n',
                encoding="utf-8",
            )

            proc = self.run_sourced(
                f"""
NNUE_NTFY=0
STOCKFISH_NET_LEDGER={shlex.quote(str(ledger))}
STOCKFISH_NET=nn-stockfish.nnue
STOCKFISH_NET_GAMES=500
STOCKFISH_NET_MIN_DELTA=-10
PROMOTED_NET_LINK={shlex.quote(str(tmp / "candidate.net"))}
QSPRT=qsprt_mock
TRACE={shlex.quote(str(trace))}
ln -s champion.nn "$PROMOTED_NET_LINK"
qsprt_mock() {{
  printf '%s %s\n' "$GAMES" "$*" >> "$TRACE"
  printf '{{"candidate":"challenger","engine":"engine","reference":"nn-stockfish.nnue","requested_games":500,"games":500,"elo":-180.8,"ci":28.5,"llr":-8.76,"llr_bound":690.78,"los":0.0,"draw":35.4}}\n' >> "$STOCKFISH_NET_LEDGER"
}}
stockfish_net_gate challenger
"""
            )

            self.assertEqual("", proc.stderr)
            self.assertEqual(0, proc.returncode)
            self.assertEqual(
                "500 --candidate challenger.nn\n",
                trace.read_text(encoding="utf-8"),
            )
            self.assertIn("delta=-6.3, upper90=19.9", proc.stdout)

    def test_stockfish_net_gate_vetoes_regression_and_preserves_champion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            ledger = tmp / "stockfish-net.jsonl"
            ledger.write_text(
                '{"candidate":"champion","engine":"engine","reference":"nn-stockfish.nnue",'
                '"requested_games":500,"games":500,"elo":-158.9,"ci":28.4,'
                '"llr":-7.84,"llr_bound":690.78,"los":0.0,"draw":34.4}\n',
                encoding="utf-8",
            )

            proc = self.run_sourced(
                f"""
NNUE_NTFY=0
STOCKFISH_NET_LEDGER={shlex.quote(str(ledger))}
STOCKFISH_NET=nn-stockfish.nnue
STOCKFISH_NET_GAMES=500
PROMOTED_NET_LINK={shlex.quote(str(tmp / "candidate.net"))}
QSPRT=qsprt_mock
ln -s champion.nn "$PROMOTED_NET_LINK"
qsprt_mock() {{
  printf '{{"candidate":"challenger","engine":"engine","reference":"nn-stockfish.nnue","requested_games":500,"games":500,"elo":-197.4,"ci":29.8,"llr":-9.15,"llr_bound":690.78,"los":0.0,"draw":31.8}}\\n' >> "$STOCKFISH_NET_LEDGER"
}}
if stockfish_net_gate challenger; then
  exit 9
fi
printf '%s %s\\n' "$stockfish_delta" "$stockfish_upper90"
[[ $(readlink "$PROMOTED_NET_LINK") == champion.nn ]]
"""
            )

            self.assertEqual("", proc.stderr)
            self.assertEqual(0, proc.returncode)
            self.assertIn("-38.5 -11.6", proc.stdout)
            self.assertEqual("champion.nn", os.readlink(tmp / "candidate.net"))

    def test_stockfish_net_gate_propagates_benchmark_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            proc = self.run_sourced(
                f"""
NNUE_NTFY=0
STOCKFISH_NET_LEDGER={shlex.quote(str(tmp / "stockfish-net.jsonl"))}
QSPRT=qsprt_mock
qsprt_mock() {{ return 9; }}
stockfish_net_gate challenger
"""
            )

            self.assertEqual(1, proc.returncode)
            self.assertIn(
                "FAILED: Stockfish net benchmark failed for challenger",
                proc.stderr,
            )

    def test_plan_uses_training_build_without_validation_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            home = tmp / "home"
            home.mkdir()
            build = tmp / "build.json"
            arch = tmp / "architecture.json"
            defaults = tmp / "defaults.json"
            captured = tmp / "captured-build.json"
            helper = tmp / "train"
            build.write_text(
                json.dumps({
                    "run": "native-2.0.0-rc1",
                    "lineage": "native",
                    "continue_from": "native-2.0.0-rc0",
                    "reference": "~/assets/nets/nn-0ee0657fb25e.nnue",
                    "hypothesis": "fresh scratch candidate",
                    "data": {
                        "source_binpack": "data/stockfish/master-binpacks/farseerT76.binpack",
                        "limit": 100000000,
                        "offset": 0,
                    },
                }, indent=2) + "\n",
                encoding="utf-8",
            )
            arch.write_text('{"input_buckets":16}\n', encoding="utf-8")
            defaults.write_text("{}\n", encoding="utf-8")
            helper.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "build=\n"
                "while (($#)); do\n"
                "  case \"$1\" in\n"
                "    --build) build=$2; shift 2 ;;\n"
                "    *) shift ;;\n"
                "  esac\n"
                "done\n"
                "cp \"$build\" \"$CAPTURED_BUILD\"\n",
                encoding="utf-8",
            )
            helper.chmod(0o755)

            proc = subprocess.run(
                [str(REPO / "nnue"), "plan"],
                cwd=tmp,
                env={
                    **os.environ,
                    "BUILD": str(build),
                    "ARCH": str(arch),
                    "DEFAULTS": str(defaults),
                    "HOME": str(home),
                    "TRAIN_HELPER": str(helper),
                    "ALLOW_STALE_TRAIN_HELPER": "1",
                    "NNUE_NTFY": "0",
                    "CAPTURED_BUILD": str(captured),
                },
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            self.assertEqual("", proc.stderr)
            resolved = json.loads(captured.read_text(encoding="utf-8"))
            self.assertNotIn("reference", resolved)
            self.assertEqual("native-2.0.0-rc0", resolved["continue_from"])

    def test_iteration_commit_keeps_accepted_build_and_leaves_next_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            build = tmp / "build.json"
            arch = tmp / "architecture.json"
            build.write_text(
                json.dumps({
                    "run": "uho-native-1.0.34",
                    "lineage": "native",
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
                    "lineage": "native",
                    "continue_from": "uho-native-1.0.33",
                    "reference": "uho-native-1.0.33",
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
            source = (REPO / "nnue").read_text(encoding="utf-8")
            harness = source.split('case "$cmd" in', 1)[0] + '''
BUILD=build.json
ARCH=architecture.json
PROMOTED_NET_LINK=candidate.net
bump_build_json "uho-native-1.0.35" "6.0" "0.49/2.20 (22%)" "forge command" "Forge result"
'''
            harness_path = tmp / "harness.sh"
            harness_path.write_text(harness, encoding="utf-8")

            subprocess.run(["bash", str(harness_path)], cwd=tmp, check=True)

            self.assertEqual(
                "uho-native-1.0.35.nn",
                os.readlink(tmp / "candidate.net"),
            )

            subject = subprocess.run(
                ["git", "show", "--no-patch", "--format=%s", "HEAD"],
                cwd=tmp,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.strip()
            self.assertEqual(
                "uho-native-1.0.35.nn: Elo 6.0,LLR 0.49/2.20 (22%)",
                subject,
            )

            body = subprocess.run(
                ["git", "show", "--no-patch", "--format=%B", "HEAD"],
                cwd=tmp,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout
            self.assertIn(
                "Hypothesis:\nContinue accepted uho-native-1.0.33 using the next UHO data window",
                body,
            )
            self.assertIn("Forge command:\nforge command", body)
            self.assertIn("Result:\nForge result", body)

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
            self.assertNotIn("reference", working_json)
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

    def test_passed_release_candidate_prepares_next_minor_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            build = tmp / "build.json"
            arch = tmp / "architecture.json"
            build.write_text(
                json.dumps({
                    "run": "native-3.0.0-rc6",
                    "lineage": "native",
                    "continue_from": "native-3.0.0-rc5",
                    "hypothesis": "previous",
                    "data": {
                        "source_binpack": "data/stockfish/master-binpacks/farseerT74.binpack",
                        "limit": 100000000,
                        "offset": 900000000,
                    },
                }, indent=2) + "\n",
                encoding="utf-8",
            )
            arch.write_text('{"output_buckets":4}\n', encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=tmp, check=True)
            subprocess.run(["git", "config", "user.name", "Petter Wahlman"], cwd=tmp, check=True)
            subprocess.run(["git", "config", "user.email", "petter@wahlman.no"], cwd=tmp, check=True)
            subprocess.run(["git", "add", "build.json", "architecture.json"], cwd=tmp, check=True)
            subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=tmp, check=True)

            build.write_text(
                json.dumps({
                    "run": "native-3.0.0-rc7",
                    "lineage": "native",
                    "continue_from": "native-3.0.0-rc5",
                    "reference": "native-3.0.0-rc5",
                    "hypothesis": "accepted four-bucket candidate",
                    "data": {
                        "source_binpack": "data/stockfish/master-binpacks/farseerT74.binpack",
                        "limit": 100000000,
                        "offset": 1000000000,
                    },
                }, indent=2) + "\n",
                encoding="utf-8",
            )
            source = (REPO / "nnue").read_text(encoding="utf-8")
            harness = source.split('case "$cmd" in', 1)[0] + """
BUILD=build.json
ARCH=architecture.json
PROMOTED_NET_LINK=candidate.net
bump_build_json "native-3.0.0-rc7" "12.3" "0.50/2.20 (23%)" "forge command" "Forge result"
"""
            harness_path = tmp / "harness.sh"
            harness_path.write_text(harness, encoding="utf-8")

            subprocess.run(["bash", str(harness_path)], cwd=tmp, check=True)

            self.assertEqual(
                "native-3.0.0-rc7.nn",
                os.readlink(tmp / "candidate.net"),
            )

            committed_json = json.loads(subprocess.run(
                ["git", "show", "HEAD:build.json"],
                cwd=tmp,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout)
            self.assertEqual("native-3.0.0-rc7", committed_json["run"])
            self.assertEqual("native-3.0.0-rc5", committed_json["continue_from"])
            self.assertEqual(1000000000, committed_json["data"]["offset"])

            working_json = json.loads(build.read_text(encoding="utf-8"))
            self.assertEqual("native-3.1.0-rc1", working_json["run"])
            self.assertEqual("native-3.0.0-rc7", working_json["continue_from"])
            self.assertNotIn("reference", working_json)
            self.assertEqual(1100000000, working_json["data"]["offset"])

    def test_failed_iteration_commit_keeps_previous_good_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            build = tmp / "build.json"
            arch = tmp / "architecture.json"
            build.write_text(
                json.dumps({
                    "run": "uho-native-1.0.35",
                    "lineage": "native",
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
                    "lineage": "native",
                    "continue_from": "uho-native-1.0.35",
                    "reference": "uho-native-1.0.35",
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
            source = (REPO / "nnue").read_text(encoding="utf-8")
            harness = source.split('case "$cmd" in', 1)[0] + """
BUILD=build.json
ARCH=architecture.json
PROMOTED_NET_LINK=candidate.net
ln -s uho-native-1.0.35.nn "$PROMOTED_NET_LINK"
continue_from=uho-native-1.0.35
fail_build_json "uho-native-1.0.36" "-25.2" "-0.32/2.20 (-15%)" "forge command" "Forge result"
"""
            harness_path = tmp / "harness.sh"
            harness_path.write_text(harness, encoding="utf-8")

            subprocess.run(["bash", str(harness_path)], cwd=tmp, check=True)

            self.assertEqual(
                "uho-native-1.0.35.nn",
                os.readlink(tmp / "candidate.net"),
            )

            subject = subprocess.run(
                ["git", "show", "--no-patch", "--format=%s", "HEAD"],
                cwd=tmp,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.strip()
            self.assertEqual(
                "uho-native-1.0.36.nn: rejected: Elo -25.2,LLR -0.32/2.20 (-15%)",
                subject,
            )

            body = subprocess.run(
                ["git", "show", "--no-patch", "--format=%B", "HEAD"],
                cwd=tmp,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout
            self.assertIn(
                "Hypothesis:\nContinue accepted uho-native-1.0.35 using the next UHO data window",
                body,
            )
            self.assertIn("Forge command:\nforge command", body)
            self.assertIn("Result:\nForge result", body)

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
            self.assertNotIn("reference", working_json)
            self.assertEqual(600000000, working_json["data"]["offset"])

    def test_failed_initialized_candidate_preserves_origin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            build = tmp / "build.json"
            arch = tmp / "architecture.json"
            build.write_text(
                json.dumps({
                    "run": "native-3.3.0-rc0",
                    "lineage": "native",
                    "reference": "native-2.1.0-rc1",
                    "initialize_from": "native-2.1.0-rc1",
                    "hypothesis": "previous initialized candidate",
                    "data": {
                        "source_binpack": "data/stockfish/master-binpacks/farseerT74.binpack",
                        "limit": 200000000,
                        "offset": 2000000000,
                    },
                }, indent=2) + "\n",
                encoding="utf-8",
            )
            arch.write_text('{"output_buckets":4}\n', encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=tmp, check=True)
            subprocess.run(["git", "config", "user.name", "Petter Wahlman"], cwd=tmp, check=True)
            subprocess.run(["git", "config", "user.email", "petter@wahlman.no"], cwd=tmp, check=True)
            subprocess.run(["git", "add", "build.json", "architecture.json"], cwd=tmp, check=True)
            subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=tmp, check=True)

            build.write_text(
                json.dumps({
                    "run": "native-3.3.0-rc1",
                    "lineage": "native",
                    "reference": "native-2.1.0-rc1",
                    "initialize_from": "native-2.1.0-rc1",
                    "hypothesis": "four output heads from one-head parent",
                    "data": {
                        "source_binpack": "data/stockfish/master-binpacks/farseerT74.binpack",
                        "limit": 200000000,
                        "offset": 2200000000,
                    },
                }, indent=2) + "\n",
                encoding="utf-8",
            )
            source = (REPO / "nnue").read_text(encoding="utf-8")
            harness = source.split('case "$cmd" in', 1)[0] + """
BUILD=build.json
ARCH=architecture.json
continue_from=
fail_build_json "native-3.3.0-rc1" "-6.1" "-0.17/2.20 (-8%)" "forge command" "Forge result"
"""
            harness_path = tmp / "harness.sh"
            harness_path.write_text(harness, encoding="utf-8")

            subprocess.run(["bash", str(harness_path)], cwd=tmp, check=True)

            committed_json = json.loads(subprocess.run(
                ["git", "show", "HEAD:build.json"],
                cwd=tmp,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout)
            self.assertEqual("native-3.3.0-rc1", committed_json["run"])
            self.assertEqual("native-2.1.0-rc1", committed_json["reference"])
            self.assertEqual("native-2.1.0-rc1", committed_json["initialize_from"])
            self.assertNotIn("continue_from", committed_json)

            working_json = json.loads(build.read_text(encoding="utf-8"))
            self.assertEqual("native-3.3.0-rc2", working_json["run"])
            self.assertEqual("native-2.1.0-rc1", working_json["reference"])
            self.assertEqual("native-2.1.0-rc1", working_json["initialize_from"])
            self.assertNotIn("continue_from", working_json)
            self.assertEqual(2400000000, working_json["data"]["offset"])

    def test_failed_scratch_rc_does_not_invent_continue_from(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            build = tmp / "build.json"
            arch = tmp / "architecture.json"
            build.write_text(
                json.dumps({
                    "run": "native-1.9.0",
                    "lineage": "native",
                    "hypothesis": "previous candidate",
                    "data": {
                        "source_binpack": "data/stockfish/master-binpacks/farseerT76.binpack",
                        "limit": 100000000,
                        "offset": 0,
                    },
                }, indent=2) + "\n",
                encoding="utf-8",
            )
            arch.write_text('{"input_buckets":16}\n', encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=tmp, check=True)
            subprocess.run(["git", "config", "user.name", "Petter Wahlman"], cwd=tmp, check=True)
            subprocess.run(["git", "config", "user.email", "petter@wahlman.no"], cwd=tmp, check=True)
            subprocess.run(["git", "add", "build.json", "architecture.json"], cwd=tmp, check=True)
            subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=tmp, check=True)

            build.write_text(
                json.dumps({
                    "run": "native-2.0.0-rc1",
                    "lineage": "native",
                    "hypothesis": "fresh scratch candidate",
                    "data": {
                        "source_binpack": "data/stockfish/master-binpacks/farseerT76.binpack",
                        "limit": 100000000,
                        "offset": 0,
                    },
                }, indent=2) + "\n",
                encoding="utf-8",
            )

            source = (REPO / "nnue").read_text(encoding="utf-8")
            harness = source.split('case "$cmd" in', 1)[0] + """
BUILD=build.json
ARCH=architecture.json
continue_from=
fail_build_json "native-2.0.0-rc1" "-25.2" "-0.32/2.20 (-15%)" "forge command" "Forge result"
"""
            harness_path = tmp / "harness.sh"
            harness_path.write_text(harness, encoding="utf-8")

            subprocess.run(["bash", str(harness_path)], cwd=tmp, check=True)

            committed_json = json.loads(subprocess.run(
                ["git", "show", "HEAD:build.json"],
                cwd=tmp,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout)
            self.assertEqual("native-2.0.0-rc1", committed_json["run"])
            self.assertNotIn("continue_from", committed_json)

            working_json = json.loads(build.read_text(encoding="utf-8"))
            self.assertEqual("native-2.0.0-rc2", working_json["run"])
            self.assertNotIn("continue_from", working_json)
            self.assertEqual(100000000, working_json["data"]["offset"])

    def test_rejected_smoke_advances_and_stops_iteration_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            build = tmp / "build.json"
            arch = tmp / "architecture.json"
            trace = tmp / "trace.txt"
            build.write_text(
                json.dumps({
                    "run": "uho-native-1.0.35",
                    "lineage": "native",
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
                    "lineage": "native",
                    "continue_from": "uho-native-1.0.35",
                    "reference": "uho-native-1.0.35",
                    "hypothesis": "candidate",
                    "data": {
                        "source_binpack": "data/nodes5000pv2_UHO.binpack",
                        "limit": 100000000,
                        "offset": 500000000,
                    },
                }, indent=2) + "\n",
                encoding="utf-8",
            )
            source = (REPO / "nnue").read_text(encoding="utf-8")
            harness = source.split('case "$cmd" in', 1)[0] + """
BUILD=build.json
ARCH=architecture.json
NNUE_NTFY=0
ITERATIONS=2
train() { printf 'train:%s\n' "$run" >> "$TRACE"; }
gates() { printf 'gates:%s\n' "$run" >> "$TRACE"; }
stockfish_net_gate() { printf 'stockfish:%s\n' "$run" >> "$TRACE"; }
sprt_gate() {
  printf 'sprt:%s\n' "$run" >> "$TRACE"
  last_sprt_elo=-25.2
  last_sprt_llr='-0.32/2.20 (-15%)'
  last_sprt_command='forge command'
  last_sprt_line='Forge smoke games=400/400 elo=-25.2 ci=28.8 llr=-0.32/2.20 (-15%) los=30.0% draw=35.0%'
  return 1
}
iterate
"""
            harness_path = tmp / "harness.sh"
            harness_path.write_text(harness, encoding="utf-8")
            env = os.environ.copy()
            env["TRACE"] = str(trace)

            proc = subprocess.run(
                ["bash", str(harness_path)],
                cwd=tmp,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(0, proc.returncode, proc.stderr)
            self.assertEqual(
                "train:uho-native-1.0.36\n"
                "gates:uho-native-1.0.36\n"
                "stockfish:uho-native-1.0.36\n"
                "sprt:uho-native-1.0.36\n",
                trace.read_text(encoding="utf-8"),
            )
            subjects = subprocess.run(
                ["git", "log", "--format=%s", "--reverse"],
                cwd=tmp,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.splitlines()
            self.assertEqual([
                "base",
                "uho-native-1.0.36.nn: rejected: Elo -25.2,LLR -0.32/2.20 (-15%)",
            ], subjects)
            working_json = json.loads(build.read_text(encoding="utf-8"))
            self.assertEqual("uho-native-1.0.37", working_json["run"])
            self.assertEqual("uho-native-1.0.35", working_json["continue_from"])
            self.assertNotIn("reference", working_json)
            self.assertEqual(600000000, working_json["data"]["offset"])

    def test_smoke_gate_rejects_bad_elo_or_negative_llr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            source = (REPO / "nnue").read_text(encoding="utf-8")
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

    def test_move_gate_skips_missing_cases_when_not_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            home = tmp / "home"
            nets = home / "assets" / "nets"
            nets.mkdir(parents=True)
            (nets / "candidate.nn").write_bytes(b"candidate")
            (nets / "reference.nn").write_bytes(b"reference")
            build = tmp / "build.json"
            build.write_text('{"run":"candidate","continue_from":"reference"}\n', encoding="utf-8")

            source = (REPO / "nnue").read_text(encoding="utf-8")
            harness = source.split('case "$cmd" in', 1)[0] + """
BUILD="$TEST_BUILD"
HOME="$TEST_HOME"
NNUE_NTFY=0
CASES="$MISSING_CASES"
MOVE_GATE_STRICT=0
move_gate
"""
            harness_path = tmp / "harness.sh"
            harness_path.write_text(harness, encoding="utf-8")
            env = os.environ.copy()
            env.update({
                "MISSING_CASES": str(tmp / "missing.jsonl"),
                "TEST_BUILD": str(build),
                "TEST_HOME": str(home),
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

            self.assertEqual("", proc.stderr)
            self.assertIn("SKIP", proc.stdout)

    def test_move_gate_fails_missing_cases_when_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            home = tmp / "home"
            nets = home / "assets" / "nets"
            nets.mkdir(parents=True)
            (nets / "candidate.nn").write_bytes(b"candidate")
            (nets / "reference.nn").write_bytes(b"reference")
            build = tmp / "build.json"
            build.write_text('{"run":"candidate","continue_from":"reference"}\n', encoding="utf-8")

            source = (REPO / "nnue").read_text(encoding="utf-8")
            harness = source.split('case "$cmd" in', 1)[0] + """
BUILD="$TEST_BUILD"
HOME="$TEST_HOME"
NNUE_NTFY=0
CASES="$MISSING_CASES"
MOVE_GATE_STRICT=1
move_gate
"""
            harness_path = tmp / "harness.sh"
            harness_path.write_text(harness, encoding="utf-8")
            env = os.environ.copy()
            env.update({
                "MISSING_CASES": str(tmp / "missing.jsonl"),
                "TEST_BUILD": str(build),
                "TEST_HOME": str(home),
            })

            proc = subprocess.run(
                ["bash", str(harness_path)],
                cwd=tmp,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertNotEqual(0, proc.returncode)
            self.assertIn("missing move gate cases", proc.stderr)

    def test_sprt_gate_runs_full_sprt_after_inconclusive_smoke(self) -> None:
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
            trace = tmp / "trace.txt"

            source = (REPO / "nnue").read_text(encoding="utf-8")
            harness = source.split('case "$cmd" in', 1)[0] + """
BUILD="$TEST_BUILD"
HOME="$TEST_HOME"
NNUE_NTFY=0
GAMES=800
run_sprt_once() {
  printf '%s:%s:%s\n' "$1" "$2" "$3" >> "$TRACE"
  if [[ "$1" == smoke ]]; then
    last_sprt_elo=4.0
    last_sprt_llr='0.10/2.20 (5%)'
    last_sprt_line='smoke inconclusive'
    return 0
  fi
  last_sprt_elo=5.0
  last_sprt_llr='0.20/2.20 (9%)'
  last_sprt_line='full positive'
}
sprt_gate
"""
            harness_path = tmp / "harness.sh"
            harness_path.write_text(harness, encoding="utf-8")
            env = os.environ.copy()
            env.update({
                "TEST_BUILD": str(build),
                "TEST_HOME": str(home),
                "TRACE": str(trace),
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

            self.assertEqual("", proc.stderr)
            self.assertEqual(
                "smoke:400:sprt_smoke\nsprt:800:sprt\n",
                trace.read_text(encoding="utf-8"),
            )

    def test_skip_smoke_runs_full_sprt_directly(self) -> None:
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
            trace = tmp / "trace.txt"

            source = (REPO / "nnue").read_text(encoding="utf-8")
            harness = source.split('case "$cmd" in', 1)[0] + """
BUILD="$TEST_BUILD"
HOME="$TEST_HOME"
NNUE_NTFY=0
SKIP_SMOKE=1
GAMES=800
run_sprt_once() {
  printf '%s:%s:%s\n' "$1" "$2" "$3" >> "$TRACE"
  [[ "$1" == sprt ]] || return 9
  last_sprt_elo=5.0
  last_sprt_llr='0.20/2.20 (9%)'
  last_sprt_line='full positive'
}
sprt_gate
"""
            harness_path = tmp / "harness.sh"
            harness_path.write_text(harness, encoding="utf-8")
            env = os.environ.copy()
            env.update({
                "TEST_BUILD": str(build),
                "TEST_HOME": str(home),
                "TRACE": str(trace),
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

            self.assertEqual("", proc.stderr)
            self.assertEqual("sprt:800:sprt\n", trace.read_text(encoding="utf-8"))

    def test_skip_smoke_rejects_failed_full_sprt(self) -> None:
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
            trace = tmp / "trace.txt"

            source = (REPO / "nnue").read_text(encoding="utf-8")
            harness = source.split('case "$cmd" in', 1)[0] + """
BUILD="$TEST_BUILD"
HOME="$TEST_HOME"
NNUE_NTFY=0
SKIP_SMOKE=1
GAMES=800
run_sprt_once() {
  printf '%s:%s:%s\n' "$1" "$2" "$3" >> "$TRACE"
  [[ "$1" == sprt ]] || return 9
  last_sprt_elo=-1.0
  last_sprt_llr='-0.01/2.20 (0%)'
  last_sprt_line='full failed'
}
sprt_gate
"""
            harness_path = tmp / "harness.sh"
            harness_path.write_text(harness, encoding="utf-8")
            env = os.environ.copy()
            env.update({
                "TEST_BUILD": str(build),
                "TEST_HOME": str(home),
                "TRACE": str(trace),
            })

            proc = subprocess.run(
                ["bash", str(harness_path)],
                cwd=tmp,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertNotEqual(0, proc.returncode)
            self.assertEqual("sprt:800:sprt\n", trace.read_text(encoding="utf-8"))

    def test_sprt_waits_matching_active_forge_run_for_build_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            home = tmp / "home"
            forge = home / "code" / "cpp" / "chess" / "forge" / "scripts" / "forge"
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

            fake_forge = tmp / "forge"
            fake_forge.write_text(
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
                "echo unexpected forge call: $* >&2\n"
                "exit 2\n",
                encoding="utf-8",
            )
            fake_forge.chmod(0o755)
            calls = tmp / "calls.txt"
            done = tmp / "done"

            source = (REPO / "nnue").read_text(encoding="utf-8")
            harness = source.split('case "$cmd" in', 1)[0] + """
NNUE_NTFY=0
SOLO=0
BUILD="$TEST_BUILD"
FORGE="$TEST_FORGE"
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
                "TEST_FORGE": str(fake_forge),
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

    def test_sprt_ignores_stale_logged_forge_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            home = tmp / "home"
            (home / "assets" / "nets").mkdir(parents=True)
            engine = home / "assets" / "engines" / "enyo_91ede5f"
            engine.parent.mkdir(parents=True)
            engine.write_text("#!/bin/sh\n", encoding="utf-8")
            engine.chmod(0o755)
            (home / "assets" / "nets" / "candidate.nn").write_bytes(b"candidate")
            (home / "assets" / "nets" / "reference.nn").write_bytes(b"reference")

            log_dir = tmp / "runs" / "candidate" / "logs"
            log_dir.mkdir(parents=True)
            (log_dir / "05-sprt-800.log").write_text(
                "run: id=candidate-sprt-800-old\n",
                encoding="utf-8",
            )
            build = tmp / "build.json"
            build.write_text(
                '{"run":"candidate","continue_from":"reference","hypothesis":"fresh after stale"}\n',
                encoding="utf-8",
            )

            fake_forge = tmp / "forge"
            fake_forge.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "if [[ \"$1\" == status && \"${2:-}\" == candidate-sprt-800-old ]]; then\n"
                "  echo 'run not found' >&2\n"
                "  exit 1\n"
                "fi\n"
                "if [[ \"$1\" == status && \"${2:-}\" == --json ]]; then\n"
                "  printf '%s\\n' '{\"runs\":[]}'\n"
                "  exit 0\n"
                "fi\n"
                "if [[ \"$1\" == sprt ]]; then\n"
                "  printf '%s\\n' \"$*\" >> \"$FORGE_CALLS\"\n"
                "  exit 0\n"
                "fi\n"
                "if [[ \"$1\" == status && \"${2:-}\" == candidate-sprt-800-* ]]; then\n"
                "  printf '%s\\n' '{\"state\":\"done\",\"progress_fields\":[\"games=800/800\",\"elo=+7.5\",\"llr=0.40/2.20 (18%)\"]}'\n"
                "  exit 0\n"
                "fi\n"
                "echo unexpected forge call: $* >&2\n"
                "exit 2\n",
                encoding="utf-8",
            )
            fake_forge.chmod(0o755)
            forge_calls = tmp / "forge-calls.txt"

            source = (REPO / "nnue").read_text(encoding="utf-8")
            harness = source.split('case "$cmd" in', 1)[0] + """
NNUE_NTFY=0
SOLO=0
BUILD="$TEST_BUILD"
FORGE="$TEST_FORGE"
HOME="$TEST_HOME"
ENGINE="$HOME/assets/engines/enyo_91ede5f"
run=candidate
continue_from=reference
reference_net="$HOME/assets/nets/reference.nn"
candidate_net="$HOME/assets/nets/candidate.nn"
log_dir="runs/candidate/logs"
run_sprt_once sprt 800 sprt
printf 'elo=%s\n' "$last_sprt_elo"
printf 'llr=%s\n' "$last_sprt_llr"
"""
            harness_path = tmp / "harness.sh"
            harness_path.write_text(harness, encoding="utf-8")
            env = os.environ.copy()
            env.update({
                "TEST_BUILD": str(build),
                "TEST_FORGE": str(fake_forge),
                "TEST_HOME": str(home),
                "FORGE_CALLS": str(forge_calls),
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

            self.assertIn("ignoring stale Forge run from log: candidate-sprt-800-old", proc.stdout)
            self.assertIn("elo=7.5", proc.stdout)
            self.assertIn("llr=0.40/2.20 (18%)", proc.stdout)
            calls = forge_calls.read_text(encoding="utf-8")
            self.assertIn("sprt --run candidate-sprt-800-", calls)
            self.assertIn("--candidate ~/assets/engines/enyo_91ede5f", calls)
            self.assertIn("--reference ~/assets/engines/enyo_91ede5f", calls)
            self.assertIn("--candidate-uci nnue_file=~/assets/nets/candidate.nn", calls)
            self.assertIn("--reference-uci nnue_file=~/assets/nets/reference.nn", calls)
            self.assertIn("--run candidate-sprt-800-", calls)
            self.assertNotIn("--run candidate-sprt-800-old", calls)

if __name__ == "__main__":
    unittest.main()
