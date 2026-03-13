#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, Iterator, List, Tuple


TripleKey = Tuple[str, str, str, str]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def iter_entries(data: Dict) -> Iterator[Tuple[TripleKey, List[int]]]:
    for direction, relations in data.items():
        if not isinstance(relations, dict):
            continue
        for relation_name, sources in relations.items():
            if not isinstance(sources, dict):
                continue
            for source_name, targets in sources.items():
                if not isinstance(targets, dict):
                    continue
                for target_name, rule_ids in targets.items():
                    yield (direction, relation_name, source_name, target_name), rule_ids


def build_entry_map(data: Dict) -> Dict[TripleKey, List[int]]:
    return dict(iter_entries(data))


def map_to_nested_json(data_map: Dict[TripleKey, List[int]]) -> Dict:
    nested: Dict = {}
    for direction, relation_name, source_name, target_name in sorted(data_map.keys()):
        nested.setdefault(direction, {})
        nested[direction].setdefault(relation_name, {})
        nested[direction][relation_name].setdefault(source_name, {})
        nested[direction][relation_name][source_name][target_name] = data_map[
            (direction, relation_name, source_name, target_name)
        ]
    return nested


def split_file(exp_path: Path, split: str) -> Path:
    new_path = exp_path / f"applied_rules_{split}.json"
    if new_path.exists():
        return new_path
    return exp_path / "explanations-processed" / f"applied_rules_{split}.json"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Save common applied-rules keys from two experiments into a new output experiment. "
            "Only common keys are kept; rules are taken from side -r (a or p)."
        )
    )
    parser.add_argument("-d", "--dataset", default="codex-m", help="Dataset folder name (default: codex-m)")
    parser.add_argument("--data_root", default="data", help="Dataset root directory")
    parser.add_argument("-a", default="expl.anyburl", help="Experiment A folder name/path")
    parser.add_argument("-p", default="expl.pyclause", help="Experiment P folder name/path")
    parser.add_argument("-o", default="expl.new", help="Output experiment folder name/path")
    parser.add_argument(
        "-r",
        "--rule-side",
        choices=["a", "p"],
        default="a",
        help="Which side provides rules for shared keys: a or p (default: a)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.data_root) / args.dataset
    a_exp = dataset_dir / args.a
    p_exp = dataset_dir / args.p
    out_exp = dataset_dir / args.o
    if not out_exp.exists():
        out_exp.mkdir(parents=True)

    print(f"A experiment: {a_exp}")
    print(f"P experiment: {p_exp}")
    print(f"Output experiment: {out_exp}")
    print(f"Rule source side (-r): {args.rule_side}")

    splits = ["train", "valid", "test"]
    summary = {}

    for split in splits:
        a_path = split_file(a_exp, split)
        p_path = split_file(p_exp, split)
        out_path = split_file(out_exp, split)

        if not a_path.exists():
            raise FileNotFoundError(f"Missing A file: {a_path}")
        if not p_path.exists():
            raise FileNotFoundError(f"Missing P file: {p_path}")

        a_data = load_json(a_path)
        p_data = load_json(p_path)

        print(f"Loaded A: {a_path} ({len(a_data)} top-level entries)")
        a_map = build_entry_map(a_data)
        p_map = build_entry_map(p_data)

        print(f"A entries: {len(a_map)}")
        common_keys = set(a_map) & set(p_map)
        rule_map = a_map if args.rule_side == "a" else p_map
        out_map = {key: rule_map[key] for key in sorted(common_keys)}

        print(f"Common keys: {len(common_keys)}")
        print(f"Rules from side {args.rule_side}: {len(out_map)}")

        out_data = map_to_nested_json(out_map)
        dump_json(out_path, out_data)

        summary[split] = {
            "a_count": len(a_map),
            "p_count": len(p_map),
            "common_count": len(common_keys),
            "saved_count": len(out_map),
            "output_file": str(out_path),
        }

    print("\n=== Done ===")
    for split in splits:
        item = summary[split]
        print(
            f"[{split}] A={item['a_count']} P={item['p_count']} "
            f"Common={item['common_count']} Saved={item['saved_count']}"
        )
        print(f"  -> {item['output_file']}")


if __name__ == "__main__":
    main()
