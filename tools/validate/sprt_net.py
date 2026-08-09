#!/usr/bin/env python3
"""Run a distributed SPRT of one net against a reference net, on a fixed engine.

The result itself is recorded centrally by forge (~/code/chess/forge/logs/sprt.db,
forge_lib.status.record_sprt_completion) for every SPRT run regardless of what
launches it. This script only launches the run; Forge owns completion."""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def check_engine_loads_net(engine: Path, role: str, net: Path) -> None:
    try:
        result = subprocess.run(
            [str(engine)],
            input=f"setoption name nnue_file value {net}\nquit\n",
            capture_output=True,
            text=True,
            timeout=20,
        )
    except subprocess.TimeoutExpired as exc:
        result = exc

    output = (result.stdout or "") + (result.stderr or "")
    resolved_net = net.resolve()
    rc = getattr(result, "returncode", 1) or 0
    if rc != 0 or f"path='{resolved_net}'" not in output or re.search(r"ERROR:|falling back", output):
        print(f"Error: engine cannot load {role}: engine={engine} net={net}", file=sys.stderr)
        print("\n".join(output.splitlines()[-40:]), file=sys.stderr)
        sys.exit(1)


def run_sprt(*, engine: Path, candidate_net: Path, reference_net: Path, games: int) -> str:
    """Launch one Forge SPRT and stream deployment output."""
    command = [
        "forge", "run", "sprt",
        "--comment", f"{candidate_net.name} vs {reference_net.name}",
        "--reference", str(engine),
        "--candidate", str(engine),
        "--reference-net", str(reference_net),
        "--candidate-net", str(candidate_net),
        "--restart", "on",
        "--games", str(games),
        "--elo0", "0", "--elo1", "10", "--alpha", "1e-300", "--beta", "1e-300",
    ]
    deploy = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = []
    assert deploy.stdout is not None
    for line in deploy.stdout:
        output.append(line)
        print(line, end="", flush=True)
    returncode = deploy.wait()
    stdout = "".join(output)
    if returncode:
        raise subprocess.CalledProcessError(returncode, command, output=stdout)

    match = re.search(r"^run: id=(\S+)", stdout, re.MULTILINE)
    if not match:
        sys.exit("Error: could not parse run id from forge run sprt output")
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate", default=os.environ.get("CANDIDATE_NET", "~/assets/nets/candidate.net")
    )
    parser.add_argument(
        "--reference", default=os.environ.get("REFERENCE_NET", "~/assets/nets/nn-0ee0657fb25e.nnue")
    )
    parser.add_argument("--engine", default=os.environ.get("ENGINE", "~/assets/engines/reference"))
    args = parser.parse_args()
    games = int(os.environ.get("GAMES", "1500"))

    candidate_net = Path(args.candidate).expanduser()
    reference_net = Path(args.reference).expanduser()
    engine = Path(args.engine).expanduser()

    if not (engine.is_file() and os.access(engine, os.X_OK)):
        sys.exit(f"Error: ENGINE is not executable: {engine}")
    if not candidate_net.is_file():
        sys.exit(f"Error: CANDIDATE_NET not found: {candidate_net}")
    if not reference_net.is_file():
        sys.exit(f"Error: REFERENCE_NET not found: {reference_net}")

    check_engine_loads_net(engine, "CANDIDATE_NET", candidate_net)
    check_engine_loads_net(engine, "REFERENCE_NET", reference_net)

    run = run_sprt(engine=engine, candidate_net=candidate_net, reference_net=reference_net, games=games)
    print(f"started={run}")


if __name__ == "__main__":
    main()
