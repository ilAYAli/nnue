#!/usr/bin/env python3
"""Run a distributed SPRT of one net against a reference net, on a fixed engine.

The result itself is recorded centrally by forge (~/code/chess/forge/logs/sprt.db,
forge_lib.status.record_sprt_completion) for every SPRT run regardless of what
launches it. This script only launches the run; Forge owns completion."""

import argparse
import json
import os
import re
import shlex
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


def existing_run_matches(
    run: str, *, candidate_net: Path, reference_net: Path, games: int
) -> bool:
    status = subprocess.run(
        ["forge", "status", run, "--json"],
        capture_output=True,
        text=True,
    )
    if status.returncode:
        return False
    try:
        payload = json.loads(status.stdout)
        command = payload["commands"][0]["command"]
        tokens = shlex.split(command)
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
        return False

    def option(name: str) -> str | None:
        try:
            return tokens[tokens.index(name) + 1]
        except (ValueError, IndexError):
            return None

    candidate = option("--candidate-net")
    reference = option("--reference-net")
    return (
        candidate is not None
        and reference is not None
        and Path(candidate).name == candidate_net.name
        and Path(reference).name == reference_net.name
        and option("--games") == str(games)
    )


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
    match = re.search(r"^run: id=(\S+)", stdout, re.MULTILINE)
    if returncode:
        if (
            match
            and "run already exists" in stdout
            and existing_run_matches(
                match.group(1),
                candidate_net=candidate_net,
                reference_net=reference_net,
                games=games,
            )
        ):
            print(f"run: identical existing run accepted: {match.group(1)}")
            return match.group(1)
        raise subprocess.CalledProcessError(returncode, command, output=stdout)

    if not match:
        sys.exit("Error: could not parse run id from forge run sprt output")
    run = match.group(1)

    wait = subprocess.run(["forge", "wait", run])
    if wait.returncode:
        raise subprocess.CalledProcessError(wait.returncode, ["forge", "wait", run])

    status = subprocess.run(
        ["forge", "status", run, "--json"], capture_output=True, text=True, check=True
    )
    try:
        progress = json.loads(status.stdout)["progress"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(f"invalid completed Forge status for {run}") from exc
    print(progress)
    return run


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
