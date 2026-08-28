#!/usr/bin/env python3
"""Fit and apply a measured LC0-root-Q -> fixed-depth Enyo calibration.

The direct root-Q logit is only a *source coordinate*.  It is deliberately
not a trainable target.  A valid artifact records a fit on deterministic
training pairs and independently checks it on a held-out split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
import sys
import time
from typing import Iterable


SCHEMA = "enyo.lc0-calibration.v1"

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / "posgen"))


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory_file_sha256(path: Path) -> str:
    """Hash a Forge inventory using the coordinator's canonical encoding."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "forge.lc0-inventory.v1":
        raise ValueError("unsupported LC0 inventory schema")
    files = payload.get("files")
    if not isinstance(files, list):
        raise ValueError("LC0 inventory has no file list")
    canonical = {
        "schema": "forge.lc0-inventory.v1",
        "files": [
            {"path": item["path"], "sha256": item["sha256"], "size": item["size"]}
            for item in files
            if isinstance(item, dict)
            and isinstance(item.get("path"), str)
            and isinstance(item.get("sha256"), str)
            and isinstance(item.get("size"), int)
        ],
    }
    if len(canonical["files"]) != len(files):
        raise ValueError("LC0 inventory has an invalid file entry")
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def canonical_source_file(input_root: Path, source_file: str) -> str:
    """Remove worker cache prefixes from deterministic record identities."""
    path = Path(source_file)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(input_root.expanduser().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def deterministic_split(source_file: str, record_index: int, *, holdout_percent: int = 20) -> str:
    if not 1 <= holdout_percent < 100:
        raise ValueError("holdout percent must be in [1, 99]")
    # This domain is deliberately distinct from the sampling predicate.
    # Reusing one hash makes all sampled rows land in the same split.
    key = f"split\0{source_file}\0{record_index}".encode()
    return "holdout" if int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 100 < holdout_percent else "fit"


def _finite_score(value: object, field: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}") from exc
    if not math.isfinite(score):
        raise ValueError(f"non-finite {field}")
    return score


def _pav(values: list[float], weights: list[int]) -> list[float]:
    """Weighted pool-adjacent-violators, returning nondecreasing values."""
    blocks: list[list[float | int]] = []  # start, end, weighted_sum, weight
    for index, (value, weight) in enumerate(zip(values, weights)):
        blocks.append([index, index, value * weight, weight])
        while len(blocks) > 1:
            left, right = blocks[-2], blocks[-1]
            if float(left[2]) / int(left[3]) <= float(right[2]) / int(right[3]):
                break
            right[0] = left[0]
            right[2] = float(left[2]) + float(right[2])
            right[3] = int(left[3]) + int(right[3])
            blocks.pop(-2)
    result = [0.0] * len(values)
    for start, end, weighted_sum, weight in blocks:
        mean = float(weighted_sum) / int(weight)
        for index in range(int(start), int(end) + 1):
            result[index] = mean
    return result


def fit_anchors(pairs: Iterable[dict], *, bins: int = 64) -> list[list[int]]:
    rows = sorted(
        (abs(_finite_score(pair["raw_score"], "raw_score")), abs(_finite_score(pair["target_score"], "target_score")))
        for pair in pairs
    )
    if len(rows) < 2:
        raise ValueError("need at least two fit pairs")
    bins = max(2, min(bins, len(rows)))
    xs: list[float] = []
    ys: list[float] = []
    weights: list[int] = []
    for bucket in range(bins):
        lo = bucket * len(rows) // bins
        hi = (bucket + 1) * len(rows) // bins
        chunk = rows[lo:hi]
        xs.append(float(median(item[0] for item in chunk)))
        ys.append(float(median(item[1] for item in chunk)))
        weights.append(len(chunk))
    ys = _pav(ys, weights)
    anchors: list[list[int]] = [[0, 0]]
    for x, y in zip(xs, ys):
        point = [int(round(x)), min(2045, max(0, int(round(y))))]
        # A zero LC0 coordinate carries no sign information.  It must remain
        # exactly [0, 0], even when its target bucket has nonzero magnitude.
        # Moving that anchor makes the artifact invalid and would fabricate a
        # nonzero label for an input score of zero.
        if point[0] == 0:
            continue
        if point[0] <= anchors[-1][0]:
            anchors[-1][1] = max(anchors[-1][1], point[1])
            continue
        anchors.append(point)
    if len(anchors) < 2:
        raise ValueError("fit has no usable source range")
    return anchors


def apply_anchors(score: float, anchors: list[list[int]]) -> int:
    sign = -1 if score < 0 else 1
    magnitude = abs(float(score))
    if magnitude == 0:
        return 0
    for index in range(1, len(anchors)):
        x0, y0 = anchors[index - 1]
        x1, y1 = anchors[index]
        if magnitude <= x1:
            fraction = (magnitude - x0) / max(1, x1 - x0)
            return sign * min(2045, int(round(y0 + fraction * (y1 - y0))))
    x0, y0 = anchors[-2]
    x1, y1 = anchors[-1]
    slope = (y1 - y0) / max(1, x1 - x0)
    return sign * min(2045, int(round(y1 + (magnitude - x1) * slope)))


def metrics(pairs: Iterable[dict], anchors: list[list[int]] | None = None) -> dict[str, float | int]:
    rows = [(_finite_score(pair["raw_score"], "raw_score"), _finite_score(pair["target_score"], "target_score")) for pair in pairs]
    if not rows:
        raise ValueError("no pairs")
    predicted = [apply_anchors(raw, anchors) if anchors is not None else raw for raw, _ in rows]
    targets = [target for _, target in rows]
    errors = [prediction - target for prediction, target in zip(predicted, targets)]
    denom = sum(target * target for target in targets)
    slope = sum(prediction * target for prediction, target in zip(predicted, targets)) / denom if denom else 0.0
    mean_p = sum(predicted) / len(predicted)
    mean_t = sum(targets) / len(targets)
    covariance = sum((p - mean_p) * (t - mean_t) for p, t in zip(predicted, targets))
    variance_p = sum((p - mean_p) ** 2 for p in predicted)
    variance_t = sum((t - mean_t) ** 2 for t in targets)
    corr = covariance / math.sqrt(variance_p * variance_t) if variance_p and variance_t else 0.0
    nonzero = [(p, t) for p, t in zip(predicted, targets) if p and t]
    sign = sum((p > 0) == (t > 0) for p, t in nonzero) / len(nonzero) if nonzero else 0.0
    return {
        "pairs": len(rows), "mae": round(sum(abs(error) for error in errors) / len(errors), 6),
        "bias": round(sum(errors) / len(errors), 6), "slope": round(slope, 6),
        "corr": round(corr, 6), "sign": round(sign, 6),
    }


def validate_artifact(artifact: dict) -> None:
    if artifact.get("schema") != SCHEMA or artifact.get("valid") is not True:
        raise ValueError("calibration artifact is not valid")
    anchors = artifact.get("anchors")
    if not isinstance(anchors, list) or len(anchors) < 2:
        raise ValueError("calibration artifact has no anchors")
    previous_x = -1
    previous_y = -1
    for point in anchors:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("invalid calibration anchor")
        x, y = point
        if not isinstance(x, int) or not isinstance(y, int) or x < 0 or y < 0 or y > 2045:
            raise ValueError("invalid calibration anchor value")
        if x <= previous_x or y < previous_y:
            raise ValueError("calibration anchors are not monotone")
        previous_x, previous_y = x, y
    if anchors[0] != [0, 0]:
        raise ValueError("calibration artifact must anchor zero at zero")
    holdout = artifact.get("holdout")
    if not isinstance(holdout, dict) or holdout.get("passed") is not True:
        raise ValueError("calibration artifact has not passed held-out validation")
    reference_target = artifact.get("reference_target")
    if (
        not isinstance(reference_target, dict)
        or not isinstance(reference_target.get("net_sha256"), str)
        or not reference_target["net_sha256"]
        or not isinstance(reference_target.get("engine_sha256"), list)
        or not reference_target["engine_sha256"]
        or reference_target.get("mode") != "search"
        or not isinstance(reference_target.get("depth"), int)
        or reference_target["depth"] < 1
    ):
        raise ValueError("calibration artifact has no reference-target provenance")


def load(path: Path) -> tuple[dict, str]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise ValueError("calibration artifact must be an object")
    validate_artifact(artifact)
    return artifact, hashlib.sha256(path.read_bytes()).hexdigest()


def fit_artifact(pairs: list[dict], *, bins: int, min_fit_pairs: int, min_holdout_pairs: int, min_improvement: float, max_slope_error: float) -> dict:
    fit = [pair for pair in pairs if pair.get("split") == "fit"]
    holdout = [pair for pair in pairs if pair.get("split") == "holdout"]
    if len(fit) < min_fit_pairs or len(holdout) < min_holdout_pairs:
        raise ValueError(f"insufficient independent calibration pairs: fit={len(fit)} holdout={len(holdout)}")
    if any(pair.get("target_mode") != "search" for pair in pairs):
        raise ValueError("calibration pairs must use the fixed-depth search target")
    target_depths = {pair.get("target_depth") for pair in pairs}
    if len(target_depths) != 1 or not isinstance(next(iter(target_depths)), int) or next(iter(target_depths)) < 1:
        raise ValueError("calibration pairs must identify exactly one positive target search depth")
    target_depth = next(iter(target_depths))
    if not any(_finite_score(pair["target_score"], "target_score") for pair in pairs):
        raise ValueError("all fixed-depth target scores are zero; refusing calibration")
    identities = []
    for pair in pairs:
        source_file = pair.get("source_file")
        record_index = pair.get("record_index")
        if not isinstance(source_file, str) or not source_file or not isinstance(record_index, int):
            raise ValueError("calibration pairs must identify source records")
        identities.append((source_file, record_index))
    if len(set(identities)) != len(identities):
        raise ValueError("calibration pairs contain duplicate source records")
    net_hashes = {str(pair.get("reference_net_sha256", "")) for pair in pairs}
    if "" in net_hashes or len(net_hashes) != 1:
        raise ValueError("calibration pairs must identify exactly one reference net SHA-256")
    engine_hashes = sorted({str(pair.get("reference_engine_sha256", "")) for pair in pairs})
    if "" in engine_hashes:
        raise ValueError("calibration pairs are missing reference engine provenance")
    anchors = fit_anchors(fit, bins=bins)
    raw = metrics(holdout)
    calibrated = metrics(holdout, anchors)
    improvement = 1.0 - float(calibrated["mae"]) / max(1e-9, float(raw["mae"]))
    passed = (
        improvement >= min_improvement
        and abs(float(calibrated["slope"]) - 1.0) <= max_slope_error
        and float(calibrated["sign"]) >= float(raw["sign"])
    )
    artifact = {
        "schema": SCHEMA, "valid": passed,
        "coordinate": "white-score, runtime-clamped and phase-normalized", "anchors": anchors,
        "reference_target": {"net_sha256": next(iter(net_hashes)), "engine_sha256": engine_hashes,
                             "mode": "search", "depth": target_depth},
        "fit": metrics(fit, anchors),
        "holdout": {"passed": passed, "raw": raw, "calibrated": calibrated,
                    "mae_improvement": round(improvement, 6),
                    "min_mae_improvement": min_improvement, "max_slope_error": max_slope_error},
        "split": "sha256(split + NUL + source_file + NUL + record_index) modulo 100; holdout below 20",
        "pairs_sha256": canonical_sha256(pairs),
    }
    return artifact


def read_pairs(paths: list[Path]) -> list[dict]:
    result: list[dict] = []
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    pair = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON") from exc
                if not isinstance(pair, dict):
                    raise ValueError(f"{path}:{number}: pair is not an object")
                result.append(pair)
    return result


def progress_line(stats: dict[str, int]) -> str:
    return (
        f"read={stats['read']} selected={stats['selected']} sampled={stats['sampled']} "
        f"target_nonzero={stats['target_nonzero']} written={stats['written']}"
    )


def validate_task_scope(
    inventory: Path | None,
    expected_inventory_digest: str | None,
    task_count: int,
    task_index: int,
) -> None:
    """Reject a multi-task launch that was given the coordinator inventory."""
    if task_count <= 0 or not 0 <= task_index < task_count:
        raise ValueError("invalid calibration task index/count")
    if not expected_inventory_digest or task_count == 1:
        return
    if inventory is None:
        raise ValueError("multi-task calibration requires a Forge task inventory")
    actual_inventory_digest = inventory_file_sha256(inventory)
    if actual_inventory_digest == expected_inventory_digest:
        raise ValueError(
            "Forge supplied the full LC0 inventory to a multi-task calibration task; "
            "task-scoped input partitioning is missing"
        )


def verify_target_engine(engine: object, net_sha256: str, *, depth: int, timeout_s: float) -> dict:
    """Require the engine to prove that the requested native net is active.

    The static ``eval`` UCI extension has returned zeros with a loaded native
    Enyo net.  Calibration therefore uses normal fixed-depth UCI search and
    checks both the engine's reported net digest and a material KQK score.
    This runs inside every Forge task before any source record is accepted.
    """
    output = "\n".join(getattr(engine, "output_history", ())).casefold()
    if "falling back" in output or "unsupported stockfish nnue architecture" in output:
        raise ValueError("target engine rejected the requested net; refusing fallback-engine calibration")
    if net_sha256.casefold() not in output:
        raise ValueError("target engine did not report the requested native net SHA-256")
    score, mate = engine.label("4k3/8/8/8/8/8/8/4KQ2 w - - 0 1", depth=depth, timeout_s=timeout_s)
    if mate is not None or score is None or abs(score) < 500:
        raise ValueError("target engine failed fixed-depth KQK preflight")
    return {"fen": "4k3/8/8/8/8/8/8/4KQ2 w - - 0 1", "score_stm": score}


def sample_pairs(args: argparse.Namespace) -> dict:
    """Write deterministic, independently split source/target evaluation pairs."""
    # Local imports keep artifact verification lightweight and avoid a module
    # cycle when label.py imports this module to apply an artifact.
    import lc0_to_jsonl
    from label import is_quiet, lc0_root_score
    from label_with_uci import EngineTimeout, UciEngine
    from lib import bullet_text

    if args.sample_modulus <= 0 or not 0 <= args.sample_remainder < args.sample_modulus:
        raise ValueError("invalid deterministic sampling modulus/remainder")
    validate_task_scope(args.inventory, args.expected_inventory_digest,
                        args.task_count, args.task_index)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    decode = lc0_to_jsonl.Stats()
    stats = {"schema": "enyo.lc0-calibration-pairs.v1", "input": str(args.input),
             "output": str(output), "engine": args.engine, "net": args.net,
             "net_option": args.net_option, "sample_modulus": args.sample_modulus,
             "sample_remainder": args.sample_remainder, "target_mode": "search",
             "target_depth": args.target_depth, "read": 0, "selected": 0,
             "sampled": 0, "written": 0, "skipped_timeout": 0, "skipped_no_score": 0,
             "skipped_invalid_root": 0, "skipped_filter": 0, "target_nonzero": 0}
    engine = UciEngine(args.engine, threads=args.threads, hash_mb=args.hash,
                       net=args.net, net_option=args.net_option)
    engine_path = Path(args.engine).expanduser()
    net_path = Path(args.net).expanduser()
    if not engine_path.is_file() or not net_path.is_file():
        engine.close()
        raise ValueError("calibration engine and net must be regular files")
    engine_sha256 = file_sha256(engine_path)
    net_sha256 = file_sha256(net_path)
    start = time.monotonic()
    next_progress = 10_000
    try:
        stats["target_preflight"] = verify_target_engine(
            engine, net_sha256, depth=args.target_depth, timeout_s=args.engine_timeout_s,
        )
        with temporary.open("w", encoding="utf-8") as handle:
            for row, ply in lc0_to_jsonl.iter_rows(
                args.input, inventory=args.inventory, shard_count=args.shard_count,
                shard_index=args.shard_index, max_records=args.max_records,
                top_policy=0, stats=decode,
            ):
                stats["read"] = decode.records
                if decode.records >= next_progress:
                    print(progress_line(stats), flush=True)
                    next_progress = (decode.records // 10_000 + 1) * 10_000
                if ply < args.min_ply or (args.quiet_only and not is_quiet(row)):
                    stats["skipped_filter"] += 1
                    continue
                stats["selected"] += 1
                source_file = canonical_source_file(args.input, str(row["source_file"]))
                record_index = int(row["record_index"])
                sample_hash = int.from_bytes(hashlib.sha256(f"sample\0{source_file}\0{record_index}".encode()).digest()[:8], "big")
                if sample_hash % args.sample_modulus != args.sample_remainder:
                    continue
                stats["sampled"] += 1
                root = lc0_root_score(row, eval_scale=args.eval_scale, value_epsilon=args.value_epsilon)
                if root is None:
                    stats["skipped_invalid_root"] += 1
                    continue
                row["score"] = root[0]
                raw_white = bullet_text.white_score_from_row(row, enyo_runtime_target=True)
                try:
                    target_stm, target_mate = engine.label(
                        row["fen"], depth=args.target_depth,
                        timeout_s=args.engine_timeout_s,
                    )
                    if target_mate is not None:
                        stats["skipped_no_score"] += 1
                        continue
                except EngineTimeout:
                    stats["skipped_timeout"] += 1
                    engine.restart()
                    continue
                if target_stm is None:
                    stats["skipped_no_score"] += 1
                    continue
                row["score"] = target_stm
                target_white = bullet_text.white_score_from_row(row, enyo_runtime_target=True)
                if target_white:
                    stats["target_nonzero"] += 1
                pair = {"source_file": source_file, "record_index": record_index,
                        "split": deterministic_split(source_file, record_index),
                        "raw_score": raw_white, "target_score": target_white,
                        "target_mode": "search", "target_depth": args.target_depth,
                        "reference_engine_sha256": engine_sha256,
                        "reference_net_sha256": net_sha256}
                handle.write(json.dumps(pair, sort_keys=True, separators=(",", ":")) + "\n")
                stats["written"] += 1
            handle.flush()
            os.fsync(handle.fileno())
        if not stats["written"]:
            raise ValueError("calibration sample is empty")
        if not stats["target_nonzero"]:
            raise ValueError("all fixed-depth target scores are zero; refusing calibration")
        stats["elapsed_s"] = round(time.monotonic() - start, 3)
        stats["decoder"] = {"files": decode.files, "invalid_records": decode.invalid_records,
                            "invalid_boards": decode.invalid_boards, "unsupported_records": decode.unsupported_records}
        os.replace(temporary, output)
        print(progress_line(stats), flush=True)
        print(json.dumps(stats, sort_keys=True))
        return stats
    finally:
        engine.close()
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--input", required=True, nargs="+", type=Path, help="pair JSONL files")
    fit.add_argument("--output", required=True, type=Path)
    fit.add_argument("--bins", type=int, default=64)
    fit.add_argument("--min-fit-pairs", type=int, default=50_000)
    fit.add_argument("--min-holdout-pairs", type=int, default=10_000)
    fit.add_argument("--min-mae-improvement", type=float, default=0.15)
    fit.add_argument("--max-slope-error", type=float, default=0.10)
    sample = sub.add_parser("sample")
    sample.add_argument("--input", required=True, type=Path)
    sample.add_argument("--inventory", type=Path)
    sample.add_argument("--output", required=True, type=Path)
    sample.add_argument("--engine", required=True)
    sample.add_argument("--net", required=True)
    sample.add_argument("--net-option", default="nnue_file")
    sample.add_argument("--threads", type=int, default=1)
    sample.add_argument("--hash", type=int, default=64)
    sample.add_argument("--engine-timeout-s", type=float, default=30.0)
    sample.add_argument("--target-depth", type=int, default=1)
    sample.add_argument("--max-records", type=int, default=0)
    sample.add_argument("--min-ply", type=int, default=16)
    sample.add_argument("--quiet-only", action=argparse.BooleanOptionalAction, default=True)
    sample.add_argument("--shard-count", type=int, default=1)
    sample.add_argument("--shard-index", type=int, default=0)
    sample.add_argument("--sample-modulus", type=int, default=10000)
    sample.add_argument("--sample-remainder", type=int, default=0)
    sample.add_argument("--expected-inventory-digest")
    sample.add_argument("--task-count", type=int, default=1)
    sample.add_argument("--task-index", type=int, default=0)
    sample.add_argument("--eval-scale", type=float, default=400.0)
    sample.add_argument("--value-epsilon", type=float, default=1e-6)
    args = parser.parse_args()
    if args.command == "sample":
        if args.target_depth < 1:
            raise ValueError("--target-depth must be positive")
        sample_pairs(args)
        return
    pairs = read_pairs(args.input)
    artifact = fit_artifact(pairs, bins=args.bins, min_fit_pairs=args.min_fit_pairs,
                            min_holdout_pairs=args.min_holdout_pairs,
                            min_improvement=args.min_mae_improvement, max_slope_error=args.max_slope_error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validate_artifact(artifact)
    print(json.dumps({"output": str(args.output), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(), "holdout": artifact["holdout"]}, sort_keys=True))


if __name__ == "__main__":
    main()
