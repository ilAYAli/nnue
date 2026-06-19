#!/usr/bin/env python3
from __future__ import annotations

import subprocess
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


if __name__ == "__main__":
    unittest.main()
