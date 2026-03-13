#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple


TripleKey = Tuple[str, str, str, str]
QueryKey = Tuple[str, str, str]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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


def list_diff(left: List[int], right: List[int]) -> List[int]:
    """Return items in left that are not matched in right (multiset difference)."""
    diff_counter = Counter(left) - Counter(right)
    result: List[int] = []
    for rule_id, count in diff_counter.items():
        result.extend([rule_id] * count)
    return sorted(result)


def aggregate_counts(entries: Iterable[TripleKey]) -> Dict[str, Counter]:
    counts = {
        "direction": Counter(),
        "relation": Counter(),
        "source": Counter(),
        "target": Counter(),
    }
    for direction, relation_name, source_name, _target_name in entries:
        counts["direction"][direction] += 1
        counts["relation"][(direction, relation_name)] += 1
        counts["source"][(direction, relation_name, source_name)] += 1
    for key in entries:
        counts["target"][key] += 1
    return counts


def compare(a_map: Dict[TripleKey, List[int]], p_map: Dict[TripleKey, List[int]]):
    a_keys = set(a_map)
    p_keys = set(p_map)

    missing_keys = sorted(a_keys - p_keys)
    common_keys = a_keys & p_keys
    extra_keys = sorted(p_keys - a_keys)

    rule_mismatches = []
    for key in sorted(common_keys):
        a_rules = a_map[key]
        p_rules = p_map[key]
        if a_rules != p_rules:
            a_only_rules = list_diff(a_rules, p_rules)
            p_only_rules = list_diff(p_rules, a_rules)
            rule_mismatches.append(
                {
                    "key": key,
                    "a_only_rules": a_only_rules,
                    "p_only_rules": p_only_rules,
                    "same_multiset": not a_only_rules and not p_only_rules,
                }
            )

    return {
        "missing_keys": missing_keys,
        "extra_keys": extra_keys,
        "rule_mismatches": rule_mismatches,
        "covers_a": not missing_keys and not rule_mismatches,
    }


def compare_query_keys(a_map: Dict[TripleKey, List[int]], p_map: Dict[TripleKey, List[int]]):
    a_query_keys = {(direction, relation_name, source_name) for direction, relation_name, source_name, _ in a_map}
    p_query_keys = {(direction, relation_name, source_name) for direction, relation_name, source_name, _ in p_map}

    missing_query_keys = sorted(a_query_keys - p_query_keys)
    extra_query_keys = sorted(p_query_keys - a_query_keys)

    return {
        "a_query_key_count": len(a_query_keys),
        "p_query_key_count": len(p_query_keys),
        "same_query_key_count": len(a_query_keys) == len(p_query_keys),
        "missing_query_keys": missing_query_keys,
        "extra_query_keys": extra_query_keys,
    }


def format_key(key: TripleKey) -> str:
    direction, relation_name, source_name, target_name = key
    return (
        f"direction={direction} relation={relation_name} "
        f"source={source_name} target={target_name}"
    )


def format_query_key(key: QueryKey) -> str:
    direction, relation_name, source_name = key
    return f"direction={direction} relation={relation_name} source={source_name}"


def print_sample(title: str, items: List, formatter, limit: int):
    print(f"{title}: {len(items)}")
    for item in items[:limit]:
        print(f"  - {formatter(item)}")
    if len(items) > limit:
        print(f"  ... {len(items) - limit} more")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare two applied_rules JSON files and verify whether P fully covers A "
            "with identical rule lists for all shared keys."
        )
    )
    parser.add_argument("-d", "--dataset", action="store", help="Name of dataset (libkge)", default="codex-m")
    parser.add_argument(
        "-a",
        default="expl.anyburl",
        help="Path to A file (default: codex-m/expl/.../applied_rules_test.json)",
    )
    parser.add_argument(
        "-p",
        default="expl.pyclause",
        help="Path to P file (default: codex-m/expl.pyclause/.../applied_rules_test.json)",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=10,
        help="How many sample mismatches/extras to print",
    )
    args = parser.parse_args()

    a_path = Path(f"/home/sy/2026/canonical/{args.dataset}/{args.a}/explanations-processed/applied_rules_test.json")
    p_path = Path(f"/home/sy/2026/canonical/{args.dataset}/{args.p}/explanations-processed/applied_rules_test.json")
    args.report_json = f"/home/sy/2026/canonical/{args.dataset}/{args.a}_vs_{args.p}.json"

    print(f"Loading A: {a_path}")
    a_data = load_json(a_path)
    print(f"Loading P: {p_path}")
    p_data = load_json(p_path)
    # 打印加载的文件的大小
    print(f"A file size: {a_path.stat().st_size / 1024:.2f} KB")
    print(f"P file size: {p_path.stat().st_size / 1024:.2f} KB")

    a_map = build_entry_map(a_data)
    p_map = build_entry_map(p_data)
    result = compare(a_map, p_map)
    query_result = compare_query_keys(a_map, p_map)

    a_counts = aggregate_counts(a_map.keys())
    p_counts = aggregate_counts(p_map.keys())

    print()
    print("=== Summary ===")
    print(f"A target-key count: {len(a_map)}")
    print(f"P target-key count: {len(p_map)}")
    print(f"P fully covers A: {result['covers_a']}")
    print(f"Missing keys from P: {len(result['missing_keys'])}")
    print(f"Rule mismatches on shared keys: {len(result['rule_mismatches'])}")
    print(f"Extra keys in P: {len(result['extra_keys'])}")

    print()
    print("=== Query-level (head/direction, relation, source) ===")
    print(f"A query-key count: {query_result['a_query_key_count']}")
    print(f"P query-key count: {query_result['p_query_key_count']}")
    print(f"Same query-key count: {query_result['same_query_key_count']}")
    print(f"Missing query keys from P: {len(query_result['missing_query_keys'])}")
    print(f"Extra query keys in P: {len(query_result['extra_query_keys'])}")

    print()
    print("=== Top-level counts ===")
    print(f"A directions: {dict(a_counts['direction'])}")
    print(f"P directions: {dict(p_counts['direction'])}")

    print()
    print_sample("Missing keys from P", result["missing_keys"], format_key, args.show)
    print_sample("Extra keys in P", result["extra_keys"], format_key, args.show)
    print_sample(
        "Missing query keys from P",
        query_result["missing_query_keys"],
        format_query_key,
        args.show,
    )
    print_sample(
        "Extra query keys in P",
        query_result["extra_query_keys"],
        format_query_key,
        args.show,
    )
    print_sample(
        "Rule mismatches",
        result["rule_mismatches"],
        lambda item: (
            f"{format_key(item['key'])} | "
            f"A_only={item['a_only_rules']} | "
            f"P_only={item['p_only_rules']}"
        ),
        args.show,
    )

    report = {
        "a_path": str(a_path),
        "p_path": str(p_path),
        "a_target_key_count": len(a_map),
        "p_target_key_count": len(p_map),
        "covers_a": result["covers_a"],
        "a_query_key_count": query_result["a_query_key_count"],
        "p_query_key_count": query_result["p_query_key_count"],
        "same_query_key_count": query_result["same_query_key_count"],
        "missing_query_keys": [list(key) for key in query_result["missing_query_keys"]],
        "extra_query_keys": [list(key) for key in query_result["extra_query_keys"]],
        "missing_keys": [list(key) for key in result["missing_keys"]],
        "extra_keys": [list(key) for key in result["extra_keys"]],
        "rule_mismatches": [
            {
                "key": list(item["key"]),
                "a_only_rules": item["a_only_rules"],
                "p_only_rules": item["p_only_rules"],
                "same_multiset": item["same_multiset"],
            }
            for item in result["rule_mismatches"]
        ],
    }
    report_path = Path(args.report_json)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print()
    print(f"Full report written to: {report_path}")


if __name__ == "__main__":
    main()