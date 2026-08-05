#!/usr/bin/env python3
"""Run a distributed SPRT of one net against a reference net, on a fixed engine.

The result itself is recorded centrally by forge (~/code/chess/forge/logs/sprt.db,
forge_lib.status.record_sprt_completion) for every SPRT run regardless of what
launches it - this script just launches one and prints the final result."""

import argparse
import json
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
    """Deploy via the async `forge run sprt` template (not the blocking `forge
    sprt` helper) with HOOK_EVENTS set, so the globally-configured
    notify_command (~/code/chess/forge/scripts/forge_event_ntfy.sh) fires real
    done/fail notifications for this run - matching how every other Forge job
    in this project reports progress, instead of silently blocking with no
    visibility. Deploy itself is async (returns as soon as workers are
    launched), so `forge resume --wait` blocks until the SPRT actually
    finishes before this function returns.
    """
    env = os.environ | {"HOOK_EVENTS": "done,fail"}
    deploy = subprocess.run(
        [
            "forge", "run", "sprt",
            "--comment", f"candidate={candidate_net.name} vs reference={reference_net.name}",
            "--reference", str(engine),
            "--candidate", str(engine),
            "--reference-net", str(reference_net),
            "--candidate-net", str(candidate_net),
            "--restart", "on",
            "--games", str(games),
            "--elo0", "0", "--elo1", "10", "--alpha", "1e-300", "--beta", "1e-300",
        ],
        env=env, capture_output=True, text=True, check=True,
    )
    print(deploy.stdout, end="")

    match = re.search(r"^run: id=(\S+)", deploy.stdout, re.MULTILINE)
    if not match:
        sys.exit("Error: could not parse run id from forge run sprt output")
    run = match.group(1)

    subprocess.run(
        ["forge", "resume", run, "--wait", "--verify", "--timeout-seconds", "0"],
        env=env, check=True,
    )
    return run


def print_result(*, run: str, requested_games: int) -> None:
    status = json.loads(subprocess.run(["forge", "status", run, "--json"], capture_output=True, text=True, check=True).stdout)

    if not status.get("completed_at"):
        sys.exit("Error: incomplete Forge result")

    metrics = {}
    for field in status.get("display", {}).get("fields", []):
        key, _, value = field.partition("=")
        metrics[key] = value

    games = int(metrics["games"].split("/")[0])
    if games != requested_games:
        sys.exit(f"Error: incomplete Forge result (games={games} requested={requested_games})")

    print(f"elo={metrics['elo']} llr={metrics['llr'].split('/')[0]}")


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
    print_result(run=run, requested_games=games)


if __name__ == "__main__":
    main()
