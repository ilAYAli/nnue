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
    position_source: str
    label_source: str
    clean_enyo_owned: bool
    init_chain: list[str]
    data_refs: list[str]
    data_sources: list[str]
    position_refs: list[str]
    label_refs: list[str]
    position_sources: list[str]
    label_sources: list[str]
    reasons: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "net": str(self.net),
            "init": self.init,
            "data": self.data,
            "position_source": self.position_source,
            "label_source": self.label_source,
            "clean_enyo_owned": self.clean_enyo_owned,
            "init_chain": self.init_chain,
            "data_refs": self.data_refs,
            "data_sources": self.data_sources,
            "position_refs": self.position_refs,
            "label_refs": self.label_refs,
            "position_sources": self.position_sources,
            "label_sources": self.label_sources,
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


def extract_selfplay_net_refs(text: str) -> list[str]:
    refs: list[str] = []
    for line in text.splitlines():
        if not any(marker in line for marker in (
            "posgen.py selfplay",
            "run_selfplay.sh",
            "fastchess",
        )):
            continue
        match = re.search(r"(?:^|\s)--nnue-file\s+(?P<path>\S+)", line)
        if match:
            refs.append(normalize_log_path(match.group("path")))
        match = re.search(r"(?:^|\s)option\.nnue_file=(?P<path>\S+)", line)
        if match:
            refs.append(normalize_log_path(match.group("path")))
    return refs


def extract_label_engine_refs(text: str) -> list[str]:
    refs: list[str] = []
    for line in text.splitlines():
        if not any(marker in line for marker in (
            "tools/score/score.py",
            "label_with_uci.py",
            "lc0_oracle_child_targets.py",
        )):
            continue
        for option in ("--engine", "--oracle"):
            match = re.search(rf"(?:^|\s){option}\s+(?P<path>\S+)", line)
            if match:
                refs.append(normalize_log_path(match.group("path")))
    return refs


def collect_json_refs(doc: Any) -> tuple[list[str], list[str]]:
    position_refs: list[str] = []
    label_refs: list[str] = []
    if isinstance(doc, dict):
        for key, value in doc.items():
            if key in {"source", "imported_from"} and isinstance(value, str):
                position_refs.append(value)
            elif key in {"input"} and isinstance(value, str):
                position_refs.append(value)
            elif key in {"engine", "oracle", "teacher"} and isinstance(value, str):
                label_refs.append(value)
            elif key == "source_map" and isinstance(value, dict):
                label_refs.extend(str(source) for source in value)
            else:
                child_positions, child_labels = collect_json_refs(value)
                position_refs.extend(child_positions)
                label_refs.extend(child_labels)
    elif isinstance(doc, list):
        for item in doc:
            child_positions, child_labels = collect_json_refs(item)
            position_refs.extend(child_positions)
            label_refs.extend(child_labels)
    return position_refs, label_refs


def extract_source_map_refs(text: str) -> list[str]:
    refs: list[str] = []
    for line in text.splitlines():
        if "source_map=" not in line:
            continue
        _, _, source_map = line.partition("source_map=")
        refs.extend(re.findall(r"['\"]([^'\"]+)['\"]\s*:", source_map))
    return refs


def collect_context_refs(context: RunContext) -> tuple[list[str], list[str]]:
    position_refs = extract_pattern_refs(context.text, DATA_PATTERNS)
    position_refs.extend(extract_input_data_refs(context.text))
    position_refs.extend(extract_selfplay_net_refs(context.text))
    label_refs = extract_source_map_refs(context.text)
    label_refs.extend(extract_label_engine_refs(context.text))
    for path, doc in context.json_docs:
        if path.name == "manifest.json":
            continue
        else:
            doc_positions, doc_labels = collect_json_refs(doc)
            position_refs.extend(doc_positions)
            label_refs.extend(doc_labels)
    return position_refs, label_refs


def expand_refs(
    position_refs: list[str],
    label_refs: list[str],
    *,
    max_depth: int = 2,
) -> tuple[list[str], list[str]]:
    all_positions = list(dict.fromkeys(ref for ref in position_refs if ref))
    all_labels = list(dict.fromkeys(ref for ref in label_refs if ref))
    seen = set(all_positions)
    frontier = list(all_positions)

    for _ in range(max_depth):
        new_positions: list[str] = []
        new_labels: list[str] = []
        for ref in frontier:
            path = expand_path(ref)
            if not path.exists():
                continue
            context = context_for(path)
            context_positions, context_labels = collect_context_refs(context)
            new_positions.extend(context_positions)
            new_labels.extend(context_labels)
        for ref in new_labels:
            if ref and ref not in all_labels:
                all_labels.append(ref)
        frontier = []
        for ref in new_positions:
            if ref and ref not in seen:
                seen.add(ref)
                all_positions.append(ref)
                frontier.append(ref)
    return all_positions, all_labels


def classify_position_ref(ref: str) -> set[str]:
    lower = ref.lower()
    tags: set[str] = set()
    if "berserk" in lower:
        tags.add("berserk")
    if "default.net" in lower:
        tags.add("enyo-default")
    if "lc0" in lower or "leela" in lower:
        tags.add("lc0")
    if "test80" in lower or "sfbinpack" in lower:
        tags.add("external-stockfish")
    if "lichess_eval" in lower:
        tags.add("lichess-eval")
    if "replay" in lower or "loss_replay" in lower or "logs/loss" in lower:
        tags.add("enyo-replay")
    if "selfplay" in lower or "fastchess" in lower or "/posgen/" in lower:
        tags.add("enyo-selfplay")
    return tags


def classify_label_ref(ref: str) -> set[str]:
    lower = ref.lower()
    tags: set[str] = set()
    if "lc0" in lower or "leela" in lower:
        tags.add("lc0")
    if "lichess_eval" in lower:
        tags.add("lichess-eval")
    if "stockfish" in lower or "/sf_" in lower or "sf_" in lower:
        tags.add("stockfish")
    if "enyo" in lower or "reference" in lower or "candidate" in lower:
        tags.add("enyo")
    return tags


def classify_source(tags: set[str], *, kind: str) -> str:
    if not tags:
        return "unknown"
    if kind == "label":
        if tags == {"stockfish"}:
            return "stockfish-oracle"
        if tags == {"enyo"}:
            return "enyo"
        if "lc0" in tags:
            return "lc0" if len(tags) == 1 else "mixed"
        if len(tags) == 1:
            return next(iter(tags))
        return "mixed"
    if tags == {"enyo-selfplay"}:
        return "enyo-selfplay"
    if tags == {"enyo-replay"}:
        return "enyo-replay"
    if "lc0" in tags:
        return "lc0" if len(tags) == 1 else "mixed"
    if tags <= {"enyo-selfplay", "enyo-replay", "enyo-default"}:
        return "enyo-games"
    if tags == {"external-stockfish"}:
        return "external-stockfish"
    return "mixed"


def path_is_berserk(ref: str) -> bool:
    return "berserk" in Path(ref).name.lower()


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

    position_refs: list[str] = []
    label_refs: list[str] = []
    for context in contexts:
        context_positions, context_labels = collect_context_refs(context)
        position_refs.extend(context_positions)
        label_refs.extend(context_labels)

    deduped_positions, deduped_labels = expand_refs(position_refs, label_refs)
    position_sources: set[str] = set()
    label_sources: set[str] = set()
    for ref in deduped_positions:
        position_sources.update(classify_position_ref(ref))
    for ref in deduped_labels:
        label_sources.update(classify_label_ref(ref))

    position_source = classify_source(position_sources, kind="position")
    label_source = classify_source(label_sources, kind="label")
    data_sources = set(position_sources)
    data_sources.update(label_sources)
    data = (
        f"{position_source}+{label_source}"
        if position_source != "unknown" or label_source != "unknown"
        else "unknown"
    )

    if position_source == "unknown":
        reasons.append("no position source provenance found")
    if label_source == "unknown":
        reasons.append("no label source provenance found")
    if "external-stockfish" in position_sources:
        reasons.append("position source includes external Stockfish/test80 rows")
    if "berserk" in position_sources:
        reasons.append("position source uses Berserk net during generation")
    if "stockfish" in label_sources:
        reasons.append("label source includes Stockfish oracle rows")
    if "lc0" in position_sources or "lc0" in label_sources:
        reasons.append("source includes LC0/Leela rows")
    if "lichess-eval" in data_sources:
        reasons.append("data source includes lichess-eval rows")
    if init != "random":
        reasons.append(f"init is {init}, not random")

    clean_positions = position_sources <= {
        "enyo-selfplay",
        "enyo-replay",
        "enyo-default",
    }
    clean_labels = label_sources <= {"stockfish", "enyo"}
    clean = (
        init == "random"
        and clean_positions
        and clean_labels
        and bool(position_sources)
        and bool(label_sources)
    )
    if not clean and not any(reason.startswith("clean_enyo_owned") for reason in reasons):
        reasons.append(
            "clean_enyo_owned requires random init, Enyo positions, "
            "and Stockfish/Enyo labels only"
        )

    return Provenance(
        net=net,
        init=init,
        data=data,
        position_source=position_source,
        label_source=label_source,
        clean_enyo_owned=clean,
        init_chain=chain,
        data_refs=list(dict.fromkeys([*deduped_positions, *deduped_labels])),
        data_sources=sorted(data_sources) if data_sources else ["unknown"],
        position_refs=deduped_positions,
        label_refs=deduped_labels,
        position_sources=sorted(position_sources) if position_sources else ["unknown"],
        label_sources=sorted(label_sources) if label_sources else ["unknown"],
        reasons=list(dict.fromkeys(reasons)),
    )


def print_text(provenance: Provenance) -> None:
    print(f"net={provenance.net}")
    print(f"init={provenance.init}")
    print(f"data={provenance.data}")
    print(f"position_source={provenance.position_source}")
    print(f"label_source={provenance.label_source}")
    print(f"position_sources={','.join(provenance.position_sources)}")
    print(f"label_sources={','.join(provenance.label_sources)}")
    print(f"data_sources={','.join(provenance.data_sources)}")
    print(f"clean_enyo_owned={'yes' if provenance.clean_enyo_owned else 'no'}")
    print("init_chain:")
    for item in provenance.init_chain:
        print(f"  - {item}")
    print("position_refs:")
    for item in provenance.position_refs[:50]:
        print(f"  - {item}")
    if len(provenance.position_refs) > 50:
        print(f"  - ... {len(provenance.position_refs) - 50} more")
    print("label_refs:")
    for item in provenance.label_refs[:50]:
        print(f"  - {item}")
    if len(provenance.label_refs) > 50:
        print(f"  - ... {len(provenance.label_refs) - 50} more")
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
