import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "bullet" / "distributed_smoke.py"
SPEC = importlib.util.spec_from_file_location("distributed_smoke", SCRIPT)
distributed_smoke = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(distributed_smoke)


class DistributedSmokeTests(unittest.TestCase):
    def test_remote_path_preserves_home_expansion(self) -> None:
        self.assertEqual("$HOME/code/chess/nnue", distributed_smoke.remote_path("~/code/chess/nnue"))
        self.assertEqual("/srv/nnue", distributed_smoke.remote_path("/srv/nnue"))

    def test_worker_command_has_required_membership_environment(self) -> None:
        command = distributed_smoke.build_run_command(
            role="worker",
            run="enyo-99.0.0-rc1",
            node="pwa-hak",
            data="/data/hak.bullet",
            build="build.json",
            coordinator="pwa-llm",
            port=9219,
            peers=2,
            sync_every=1,
            timeout=180,
        )
        self.assertIn("ENYO_BULLET_DISTRIBUTED_ROLE=worker", command)
        self.assertIn("ENYO_BULLET_DISTRIBUTED_NODE_ID=pwa-hak", command)
        self.assertIn("ENYO_BULLET_DISTRIBUTED_COORDINATOR_ADDR=pwa-llm:9219", command)
        self.assertIn("target/release/train run --build build.json", command)

    def test_one_worker_is_a_valid_membership_size(self) -> None:
        distributed_smoke.validate_workers("pwa-llm", ["pwa-hak"])

    def test_duplicate_or_coordinator_worker_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            distributed_smoke.validate_workers("pwa-llm", ["pwa-hak", "pwa-hak"])
        with self.assertRaises(SystemExit):
            distributed_smoke.validate_workers("pwa-llm", ["pwa-llm"])


if __name__ == "__main__":
    unittest.main()
