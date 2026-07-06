#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = (
    "runs/lichess-inacc-miss-blunder-20260612/"
    "native760_vs_native123_net_value_bad_move_gate.jsonl"
)
SCORE_RE = re.compile(r"score cp (-?\d+)")
STATIC_RE = re.compile(r"^(mae|sign|corr|bias|slope)=(-?\d+(?:\.\d+)?)(%)?$")
SPRT_RE = re.compile(
    r"Elo\s+([+-]?(?:inf|\d+(?:\.\d+)?|-inf)).*draw\s+(\d+(?:\.\d+)?)%"
)


@dataclass
class Stage:
    name: str
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    log: str | None = None
    error: str | None = None


class CandidateFailure(RuntimeError):
    pass


def expand_path(value: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def previous_version_run(run: str) -> str | None:
    prefix, sep, version_text = run.rpartition("-v")
    if not sep or not prefix or not version_text.isdigit():
        return None
    version = int(version_text)
    if version <= 1:
        return None
    return f"{prefix}-v{version - 1:0{len(version_text)}d}"


def resolved_continue_from(build: dict[str, Any]) -> str | None:
    if "continue_from" in build:
        value = build["continue_from"]
        if value is None:
            return None
        value = str(value).strip()
        return value or None
    return previous_version_run(str(build["run"]))


def run_command(
    name: str,
    command: list[str],
    log_path: Path,
    *,
    input_text: str | None = None,
) -> tuple[int, str]:
    print()
    print(f"==== {name} ====")
    print(" ".join(command))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdin=subprocess.PIPE if input_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if input_text is not None:
            assert proc.stdin is not None
            proc.stdin.write(input_text)
            proc.stdin.close()
        lines: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            lines.append(line)
        return proc.wait(), "".join(lines)


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise CandidateFailure(f"missing {description}: {path}")


def parse_static_metrics(output: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in output.splitlines():
        match = STATIC_RE.match(line.strip())
        if not match:
            continue
        key, value, pct = match.groups()
        number = float(value)
        metrics[key] = number if pct is None else number / 100.0
    return metrics


def parse_startpos(output: str) -> dict[str, Any]:
    scores = [int(item) for item in SCORE_RE.findall(output)]
    evaluator_lines = [
        line for line in output.splitlines()
        if "info string evaluator=" in line
    ]
    evaluator = evaluator_lines[-1] if evaluator_lines else ""
    input_match = re.search(r"input_buckets=(\d+)", evaluator)
    channel_match = re.search(r"feature_channels=(\d+)", evaluator)
    return {
        "native": "evaluator=native-nnue" in evaluator,
        "hidden_1024": "hidden=1024" in evaluator,
        "input_buckets": input_match.group(1) if input_match else None,
        "feature_channels": channel_match.group(1) if channel_match else None,
        "last_cp": scores[-1] if scores else None,
    }


def fail_if(condition: bool, message: str) -> None:
    if condition:
        raise CandidateFailure(message)


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Train/export one NNUE candidate, then run automatic gates.")
    ap.add_argument("--build", default="build.json",
                    help="Build config for this candidate.")
    ap.add_argument("--arch", default="architecture.json",
                    help="Architecture config passed to tools/bullet/train.")
    ap.add_argument("--defaults", default="defaults.json",
                    help="Training defaults passed to tools/bullet/train.")
    ap.add_argument("--engine", default="~/assets/engines/candidate",
                    help="Engine binary used for startpos, move gate, and SPRT.")
    ap.add_argument("--candidate-net",
                    help="Candidate .nn path. Defaults to ~/assets/nets/<run>.nn.")
    ap.add_argument("--reference-net",
                    help="Reference net path. Defaults to previous run, then nn-0ee0657fb25e.nnue.")
    ap.add_argument("--cases", default=DEFAULT_CASES,
                    help="Move-gate JSONL cases.")
    ap.add_argument("--rows", type=int, default=50000,
                    help="Rows used by static eval.")
    ap.add_argument("--games", type=int, default=100,
                    help="Games used if --with-sprt is passed.")
    ap.add_argument("--concurrency", type=int, default=10,
                    help="SPRT concurrency if --with-sprt is passed.")
    ap.add_argument("--threads", type=int, default=1,
                    help="Engine threads per SPRT game.")
    ap.add_argument("--hash", type=int, default=64,
                    help="Engine hash MB for move-gate eval.")
    ap.add_argument("--max-startpos-cp", type=int, default=300,
                    help="Reject if startpos depth-3 eval is outside +/- this cp.")
    ap.add_argument("--max-mae", type=float, default=1000.0,
                    help="Reject static eval at or above this MAE.")
    ap.add_argument("--min-sign", type=float, default=55.0,
                    help="Reject static eval below this sign percentage.")
    ap.add_argument("--min-corr", type=float, default=0.20,
                    help="Reject static eval below this correlation.")
    ap.add_argument("--max-abs-bias", type=float, default=300.0,
                    help="Reject static eval when abs(bias) is above this cp.")
    ap.add_argument("--min-slope", type=float, default=0.05,
                    help="Reject static eval below this slope.")
    ap.add_argument("--move-gate", choices=("auto", "on", "off"), default="auto",
                    help="auto runs the move gate only when the case file exists.")
    ap.add_argument("--move-gate-blocking", action="store_true",
                    help="Reject when move-gate counts regress. Default is report-only.")
    ap.add_argument("--with-sprt", action="store_true",
                    help="Run the SPRT smoke after other gates pass.")
    ap.add_argument("--min-sprt-elo", type=float, default=0.0,
                    help="Reject SPRT below this Elo when --with-sprt is used.")
    ap.add_argument("--min-sprt-draw", type=float, default=0.0,
                    help="Reject SPRT below this draw percentage when --with-sprt is used.")
    ap.add_argument("--force-export", action="store_true",
                    help="Pass --force to tools/bullet/train export.")
    ap.add_argument("--skip-train", action="store_true",
                    help="Only validate already exported candidate net.")
    args = ap.parse_args()

    build_path = expand_path(args.build)
    arch_path = expand_path(args.arch)
    defaults_path = expand_path(args.defaults)
    build = load_json(build_path)
    arch = load_json(arch_path)
    run = str(build["run"])
    previous = resolved_continue_from(build)
    run_dir = REPO_ROOT / "runs" / run
    logs_dir = run_dir / "logs"
    candidate_net = expand_path(args.candidate_net or f"~/assets/nets/{run}.nn")
    reference_net = expand_path(
        args.reference_net
        or (f"~/assets/nets/{previous}.nn" if previous else "~/assets/nets/nn-0ee0657fb25e.nnue")
    )
    data_cfg = build.get("data", {})
    data_file = expand_path(str(data_cfg.get("bullet_output", f"data/bullet/{run}.bullet")))
    engine = expand_path(args.engine)
    cases = expand_path(args.cases)
    summary_path = run_dir / "gate_summary.json"

    stages: list[Stage] = []
    summary: dict[str, Any] = {
        "run": run,
        "continue_from": previous,
        "engine": str(engine),
        "reference_net": str(reference_net),
        "candidate_net": str(candidate_net),
        "data": str(data_file),
        "stages": [],
    }

    def record(stage: Stage) -> None:
        stages.append(stage)
        summary["stages"] = [stage.__dict__ for stage in stages]
        write_summary(summary_path, summary)

    def checked_command(name: str, command: list[str]) -> str:
        code, output = run_command(name, command, logs_dir / f"{name}.log")
        status = "ok" if code == 0 else "failed"
        record(Stage(name, status, log=str(logs_dir / f"{name}.log")))
        if code != 0:
            raise CandidateFailure(f"{name} failed with exit code {code}")
        return output

    print(f"run={run}")
    print(f"continue_from={previous or ''}")
    print(f"reference_net={reference_net}")
    print(f"candidate_net={candidate_net}")
    print(f"data={data_file}")

    try:
        if not args.skip_train:
            base = [
                "tools/bullet/train",
                "--build", str(build_path),
                "--arch", str(arch_path),
                "--defaults", str(defaults_path),
            ]
            checked_command("plan", base[:1] + ["plan"] + base[1:])
            checked_command("data", base[:1] + ["data"] + base[1:])
            checked_command("train", base[:1] + ["run"] + base[1:])
            export_cmd = base[:1] + ["export"] + base[1:]
            if args.force_export:
                export_cmd.append("--force")
            checked_command("export", export_cmd)

        require_file(candidate_net, "candidate net")
        require_file(data_file, "Bullet data")
        require_file(engine, "engine")
        require_file(reference_net, "reference net")

        startpos_input = (
            "uci\n"
            f"setoption name nnue_file value {candidate_net}\n"
            "isready\n"
            "position startpos\n"
            "go depth 3\n"
            "quit\n"
        )
        code, output = run_command(
            "startpos", [str(engine)], logs_dir / "startpos.log",
            input_text=startpos_input)
        metrics = parse_startpos(output)
        expected_input_buckets = str(arch.get("input_buckets", 1))
        expected_channels = str(arch.get("feature_channels", 12))
        try:
            fail_if(code != 0, f"startpos engine exited with {code}")
            fail_if(not metrics["native"], "startpos did not load evaluator=native-nnue")
            fail_if(metrics["input_buckets"] != expected_input_buckets,
                    f"input_buckets {metrics['input_buckets']} != {expected_input_buckets}")
            fail_if(metrics["feature_channels"] != expected_channels,
                    f"feature_channels {metrics['feature_channels']} != {expected_channels}")
            fail_if(metrics["last_cp"] is None, "startpos produced no cp score")
            fail_if(abs(int(metrics["last_cp"])) > args.max_startpos_cp,
                    f"startpos cp {metrics['last_cp']} outside +/-{args.max_startpos_cp}")
            record(Stage("startpos", "ok", metrics, str(logs_dir / "startpos.log")))
        except CandidateFailure as exc:
            record(Stage("startpos", "failed", metrics, str(logs_dir / "startpos.log"), str(exc)))
            raise

        venv_python = REPO_ROOT / ".venv/bin/python"
        python = str(venv_python if venv_python.exists() else Path(sys.executable))
        static_cmd = [
            python, "tools/validate/eval_dataset.py",
            "--net", str(candidate_net),
            "--data", str(data_file),
            "--rows", str(args.rows),
            "--batch-size", "4096",
            "--device", "cpu",
            "--buckets",
        ]
        code, output = run_command("static_eval", static_cmd, logs_dir / "static_eval.log")
        metrics = parse_static_metrics(output)
        try:
            fail_if(code != 0, f"static_eval failed with exit code {code}")
            for key in ("mae", "sign", "corr", "bias", "slope"):
                fail_if(key not in metrics or not math.isfinite(metrics[key]),
                        f"static_eval missing finite {key}")
            fail_if(metrics["mae"] >= args.max_mae,
                    f"mae {metrics['mae']:.3f} >= {args.max_mae}")
            fail_if(metrics["sign"] * 100.0 < args.min_sign,
                    f"sign {metrics['sign'] * 100.0:.2f}% < {args.min_sign}%")
            fail_if(metrics["corr"] < args.min_corr,
                    f"corr {metrics['corr']:.6f} < {args.min_corr}")
            fail_if(abs(metrics["bias"]) > args.max_abs_bias,
                    f"bias {metrics['bias']:.3f} outside +/-{args.max_abs_bias}")
            fail_if(metrics["slope"] < args.min_slope,
                    f"slope {metrics['slope']:.6f} < {args.min_slope}")
            record(Stage("static_eval", "ok", metrics, str(logs_dir / "static_eval.log")))
        except CandidateFailure as exc:
            record(Stage("static_eval", "failed", metrics, str(logs_dir / "static_eval.log"), str(exc)))
            raise

        run_move_gate = args.move_gate == "on" or (args.move_gate == "auto" and cases.exists())
        if run_move_gate:
            require_file(cases, "move-gate cases")
            move_summary = run_dir / "move_gate.summary.json"
            move_cmd = [
                python, "tools/validate/eval_move_gate.py",
                "--cases", str(cases),
                "--engine", str(engine),
                "--reference-net", str(reference_net),
                "--candidate-net", str(candidate_net),
                "--threads", "1",
                "--hash", str(args.hash),
                "--summary-json", str(move_summary),
                "--output", str(run_dir / "move_gate.jsonl"),
            ]
            code, _output = run_command("move_gate", move_cmd, logs_dir / "move_gate.log")
            metrics = load_json(move_summary) if move_summary.exists() else {}
            try:
                fail_if(code != 0, f"move_gate failed with exit code {code}")
                cand = int(metrics.get("candidate_prefers_best", -1))
                ref = int(metrics.get("baseline_prefers_best", -1))
                fixed = int(metrics.get("fixed", -1))
                regressed = int(metrics.get("regressed", -1))
                passed_counts = cand >= ref and fixed > regressed
                metrics["blocking"] = bool(args.move_gate_blocking)
                metrics["passed_counts"] = passed_counts
                if args.move_gate_blocking:
                    fail_if(cand < ref, f"candidate_prefers_best {cand} < reference {ref}")
                    fail_if(fixed <= regressed, f"fixed {fixed} <= regressed {regressed}")
                record(Stage("move_gate", "ok", metrics, str(logs_dir / "move_gate.log")))
            except CandidateFailure as exc:
                record(Stage("move_gate", "failed", metrics, str(logs_dir / "move_gate.log"), str(exc)))
                raise
        else:
            record(Stage("move_gate", "skipped", {"reason": "disabled or cases missing"}))

        if args.with_sprt:
            sprt_cmd = [
                "sprt",
                "--concurrency", str(args.concurrency),
                "--threads", str(args.threads),
                "--reference", str(engine),
                "--candidate", str(engine),
                "--reference-uci", f"nnue_file={reference_net}",
                "--candidate-uci", f"nnue_file={candidate_net}",
                "--games", str(args.games),
            ]
            code, output = run_command("sprt", sprt_cmd, logs_dir / f"sprt_{args.games}.log")
            matches = SPRT_RE.findall(output)
            metrics = {}
            if matches:
                elo_text, draw_text = matches[-1]
                metrics["elo"] = float("-inf") if elo_text == "-inf" else (
                    float("inf") if elo_text in ("inf", "+inf") else float(elo_text))
                metrics["draw"] = float(draw_text)
            try:
                fail_if(code != 0, f"sprt failed with exit code {code}")
                fail_if("elo" not in metrics, "could not parse SPRT Elo")
                fail_if(metrics["elo"] < args.min_sprt_elo,
                        f"sprt elo {metrics['elo']} < {args.min_sprt_elo}")
                fail_if(metrics["draw"] < args.min_sprt_draw,
                        f"sprt draw {metrics['draw']}% < {args.min_sprt_draw}%")
                record(Stage("sprt", "ok", metrics, str(logs_dir / f"sprt_{args.games}.log")))
            except CandidateFailure as exc:
                record(Stage("sprt", "failed", metrics, str(logs_dir / f"sprt_{args.games}.log"), str(exc)))
                raise
        else:
            record(Stage("sprt", "skipped", {"reason": "pass --with-sprt to run"}))

    except CandidateFailure as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        write_summary(summary_path, summary)
        print()
        print(f"FAILED: {exc}")
        print(f"summary={summary_path}")
        return 1

    summary["status"] = "ok"
    write_summary(summary_path, summary)
    print()
    print("OK")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
