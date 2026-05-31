#!/usr/bin/env python3
"""Trace NNUE candidate provenance from local run logs and metadata."""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


INIT_PATTERNS = (
    re.compile(r"^init=(?P<path>\S+)"),
    re.compile(r"\binitializing from (?P<path>\S+)"),
    re.compile(r"(?:^|\s)--init-from-nn\s+(?P<path>\S+)"),
    re.compile(r"(?:^|\s)--base-net\s+(?P<path>\S+)"),
)
DATA_PATTERNS = (
    re.compile(r"^data=(?P<path>\S+)"),
    re.compile(r"(?:^|\s)--data\s+(?P<path>\S+)"),
    re.compile(r"\bloading train rows from (?P<path>.+)$"),
)


@dataclass
class RunContext:
    root: Path | None
    files: list[Path] = field(default_factory=list)
    text: str = ""
    json_docs: list[tuple[Path, dict[str, Any]]] = field(default_factory=list)


@dataclass
class Provenance:
    net: Path
    init: str
    data: str
    clean_enyo_owned: bool
    init_chain: list[str]
    data_refs: list[str]
    data_sources: list[str]
    reasons: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "net": str(self.net),
            "init": self.init,
            "data": self.data,
            "clean_enyo_owned": self.clean_enyo_owned,
            "init_chain": self.init_chain,
            "data_refs": self.data_refs,
            "data_sources": self.data_sources,
            "reasons": self.reasons,
        }


def expand_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def find_run_root(path: Path) -> Path | None:
    current = path if path.is_dir() else path.parent
    for parent in (current, *current.parents):
        if (parent / "runs.log").exists() or (parent / "events.jsonl").exists():
            return parent
        if parent.name == "runs":
            break
    return None


def context_for(path: Path) -> RunContext:
    root = find_run_root(path)
    search_roots: list[Path] = []
    current = path if path.is_dir() else path.parent
    search_roots.append(current)
    search_roots.extend(parent for parent in current.parents[:4])
    if root is not None:
        search_roots.append(root)

    files: list[Path] = []
    for base in dict.fromkeys(search_roots):
        for name in ("runs.log", "train.log", "config.json", "manifest.json", "meta.json"):
            candidate = base / name
            if candidate.exists() and candidate not in files:
                files.append(candidate)
        for rel in ("pack/train/meta.json", "score/meta.json"):
            candidate = base / rel
            if candidate.exists() and candidate not in files:
                files.append(candidate)
        for candidate in sorted(base.glob("assets/*/meta.json")):
            if candidate not in files:
                files.append(candidate)

    text_parts: list[str] = []
    json_docs: list[tuple[Path, dict[str, Any]]] = []
    for file in files:
        if file.suffix == ".json":
            doc = read_json(file)
            if isinstance(doc, dict):
                json_docs.append((file, doc))
            text_parts.append(read_text(file))
        else:
            text_parts.append(read_text(file))
    return RunContext(root=root, files=files, text="\n".join(text_parts), json_docs=json_docs)


def normalize_log_path(value: str) -> str:
    value = value.strip().strip("\"'")
    if " at skip=" in value:
        value = value.split(" at skip=", 1)[0]
    return value.strip()


def extract_pattern_refs(text: str, patterns: tuple[re.Pattern[str], ...]) -> list[str]:
    refs: list[str] = []
    for line in text.splitlines():
        for pattern in patterns:
            match = pattern.search(line)
            if match:
                refs.append(normalize_log_path(match.group("path")))
    return refs


def extract_bullet_weight_init(text: str) -> list[str]:
    refs: list[str] = []
    for line in text.splitlines():
        if "enyo_nn_to_bullet_weights.py" not in line:
            continue
        match = re.search(r"(?:^|\s)--input\s+(?P<path>\S+)", line)
        if match:
            refs.append(normalize_log_path(match.group("path")))
    return refs


def extract_input_data_refs(text: str) -> list[str]:
    refs: list[str] = []
    for line in text.splitlines():
        if not any(marker in line for marker in (
            "jsonl_to_bullet_text.py",
            "tools/score/score.py",
            "pack_dataset.py",
        )):
            continue
        match = re.search(r"(?:^|\s)--input\s+(?P<path>\S+)", line)
        if match:
            refs.append(normalize_log_path(match.group("path")))
    return refs


def collect_json_refs(doc: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(doc, dict):
        for key, value in doc.items():
            if key in {"source", "imported_from"} and isinstance(value, str):
                refs.append(value)
            elif key == "source_map" and isinstance(value, dict):
                refs.extend(str(source) for source in value)
            else:
                refs.extend(collect_json_refs(value))
    elif isinstance(doc, list):
        for item in doc:
            refs.extend(collect_json_refs(item))
    return refs


def extract_source_map_refs(text: str) -> list[str]:
    refs: list[str] = []
    for line in text.splitlines():
        if "source_map=" not in line:
            continue
        _, _, source_map = line.partition("source_map=")
        refs.extend(re.findall(r"['\"]([^'\"]+)['\"]\s*:", source_map))
    return refs


def collect_context_data_refs(context: RunContext) -> list[str]:
    refs = extract_pattern_refs(context.text, DATA_PATTERNS)
    refs.extend(extract_input_data_refs(context.text))
    refs.extend(extract_source_map_refs(context.text))
    for path, doc in context.json_docs:
        if path.name == "manifest.json":
            continue
        else:
            refs.extend(collect_json_refs(doc))
    return refs


def expand_data_refs(refs: list[str], *, max_depth: int = 2) -> list[str]:
    all_refs = list(dict.fromkeys(ref for ref in refs if ref))
    seen = set(all_refs)
    frontier = list(all_refs)

    for _ in range(max_depth):
        new_refs: list[str] = []
        for ref in frontier:
            path = expand_path(ref)
            if not path.exists():
                continue
            context = context_for(path)
            new_refs.extend(collect_context_data_refs(context))
        frontier = []
        for ref in new_refs:
            if ref and ref not in seen:
                seen.add(ref)
                all_refs.append(ref)
                frontier.append(ref)
    return all_refs


def classify_data_ref(ref: str) -> set[str]:
    lower = ref.lower()
    tags: set[str] = set()
    if "lc0" in lower or "leela" in lower:
        tags.add("lc0")
    if "lichess_eval" in lower:
        tags.add("lichess-eval")
    elif "lichess" in lower:
        tags.add("lichess")
    if "stockfish" in lower or "sfbinpack" in lower or "/sf_" in lower or "sf_" in lower:
        tags.add("stockfish")
    if "replay" in lower or "loss_replay" in lower or "logs/loss" in lower:
        tags.add("enyo-replay")
    if "selfplay" in lower or "fastchess" in lower:
        tags.add("enyo-selfplay")
    return tags


def classify_data(tags: set[str]) -> str:
    if not tags:
        return "unknown"
    if tags == {"enyo-selfplay"}:
        return "enyo-selfplay"
    if tags == {"enyo-replay"}:
        return "enyo-replay"
    if "lc0" in tags:
        return "lc0" if len(tags) == 1 else "mixed"
    if tags == {"stockfish"}:
        return "stockfish-labeled"
    if tags <= {"enyo-selfplay", "enyo-replay"}:
        return "enyo-games"
    return "mixed"


def path_is_berserk(ref: str) -> bool:
    return "berserk" in ref.lower()


def looks_random_init(context: RunContext) -> bool:
    text = context.text.lower()
    markers = (
        "start bullet_train",
        "training preamble",
        "enyo_l0_stdev",
        "enyo_l0-std",
        "enyo_l1_stdev",
        "enyo_l1-std",
    )
    return any(marker in text for marker in markers)


def trace_init(path: Path, *, max_depth: int = 16) -> tuple[str, list[str], list[str], list[RunContext]]:
    chain = [str(path)]
    reasons: list[str] = []
    contexts: list[RunContext] = []
    current = path
    seen = {str(current)}

    for _ in range(max_depth):
        if path_is_berserk(str(current)):
            reasons.append(f"init chain reaches Berserk: {current}")
            return "berserk-derived", chain, reasons, contexts

        context = context_for(current)
        contexts.append(context)
        init_refs = extract_pattern_refs(context.text, INIT_PATTERNS)
        init_refs.extend(extract_bullet_weight_init(context.text))

        if not init_refs:
            if looks_random_init(context):
                return "random", chain, reasons, contexts
            reasons.append(f"no init provenance found for {current}")
            return "unknown", chain, reasons, contexts

        next_ref = init_refs[-1]
        chain.append(next_ref)
        if path_is_berserk(next_ref):
            reasons.append(f"init chain reaches Berserk: {next_ref}")
            return "berserk-derived", chain, reasons, contexts

        next_path = expand_path(next_ref)
        key = str(next_path)
        if key in seen:
            reasons.append(f"init chain loop at {next_path}")
            return "unknown", chain, reasons, contexts
        seen.add(key)
        current = next_path

    reasons.append(f"init chain exceeded {max_depth} steps")
    return "unknown", chain, reasons, contexts


def analyze(path: Path) -> Provenance:
    net = expand_path(path)
    init, chain, reasons, contexts = trace_init(net)

    data_refs: list[str] = []
    for context in contexts:
        data_refs.extend(collect_context_data_refs(context))

    deduped_refs = expand_data_refs(data_refs)
    data_sources: set[str] = set()
    for ref in deduped_refs:
        data_sources.update(classify_data_ref(ref))
    data = classify_data(data_sources)

    if data == "unknown":
        reasons.append("no data source provenance found")
    if "stockfish" in data_sources:
        reasons.append("data source includes Stockfish-labeled/generated rows")
    if "lc0" in data_sources:
        reasons.append("data source includes LC0/Leela rows")
    if "lichess-eval" in data_sources:
        reasons.append("data source includes lichess-eval rows")
    if init != "random":
        reasons.append(f"init is {init}, not random")

    clean_sources = data_sources <= {"enyo-selfplay", "enyo-replay"}
    clean = init == "random" and clean_sources and bool(data_sources)
    if not clean and not any(reason.startswith("clean_enyo_owned") for reason in reasons):
        reasons.append("clean_enyo_owned requires random init and only Enyo selfplay/replay data")

    return Provenance(
        net=net,
        init=init,
        data=data,
        clean_enyo_owned=clean,
        init_chain=chain,
        data_refs=deduped_refs,
        data_sources=sorted(data_sources) if data_sources else ["unknown"],
        reasons=list(dict.fromkeys(reasons)),
    )


def print_text(provenance: Provenance) -> None:
    print(f"net={provenance.net}")
    print(f"init={provenance.init}")
    print(f"data={provenance.data}")
    print(f"data_sources={','.join(provenance.data_sources)}")
    print(f"clean_enyo_owned={'yes' if provenance.clean_enyo_owned else 'no'}")
    print("init_chain:")
    for item in provenance.init_chain:
        print(f"  - {item}")
    print("data_refs:")
    for item in provenance.data_refs[:50]:
        print(f"  - {item}")
    if len(provenance.data_refs) > 50:
        print(f"  - ... {len(provenance.data_refs) - 50} more")
    print("reasons:")
    for item in provenance.reasons:
        print(f"  - {item}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net", required=True, help="Candidate .nn file.")
    parser.add_argument(
        "--require-clean-enyo-owned",
        action="store_true",
        help="Exit nonzero unless provenance is random-init and Enyo-only data.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    provenance = analyze(expand_path(args.net))
    if args.json:
        print(json.dumps(provenance.as_dict(), indent=2, sort_keys=True))
    else:
        print_text(provenance)
    if args.require_clean_enyo_owned and not provenance.clean_enyo_owned:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
