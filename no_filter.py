#!/usr/bin/env python3
import argparse
import os
import re
from collections import defaultdict


def read_ids(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read().splitlines()
    return [line.split("\t")[1] for line in raw]


def _split_rule_line(line: str):
    parts = line.rstrip("\n").split("\t")
    if len(parts) >= 4:
        return parts
    return re.split(r"\s+", line.strip(), maxsplit=3)


def extract_head_relation(rule_body: str):
    head = rule_body.split("<=")[0].strip()
    match = re.match(r"^\s*([^\(]+)\(", head)
    if not match:
        return None
    return match.group(1).strip()


def parse_rule_file_metadata(rule_file, relation_ids):
    relation_to_id = {rel: idx for idx, rel in enumerate(relation_ids)}
    rule_relation_by_id = {}

    with open(rule_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            parts = _split_rule_line(line)
            if len(parts) < 4:
                continue
            rel = extract_head_relation(parts[3].strip())
            rel_id = relation_to_id.get(rel)
            if rel_id is not None:
                rule_relation_by_id[int(line_no)] = int(rel_id)
    return rule_relation_by_id


def parse_raw_dependency_file(path, rule_relation_by_id):
    pairs_by_relation = defaultdict(list)
    seen_pairs = defaultdict(set)

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                parts = re.split(r"\s+", line)
            if len(parts) < 5:
                continue

            try:
                id1 = int(parts[3])
                id2 = int(parts[4])
            except Exception:
                continue

            rel1 = rule_relation_by_id.get(id1)
            rel2 = rule_relation_by_id.get(id2)
            if rel1 is None or rel2 is None or rel1 != rel2:
                continue

            a, b = (id1, id2) if id1 <= id2 else (id2, id1)
            if (a, b) in seen_pairs[rel1]:
                continue
            seen_pairs[rel1].add((a, b))
            pairs_by_relation[int(rel1)].append((a, b))

    return dict(pairs_by_relation)


def write_pairs(input_path, output_path, rule_relation_by_id):
    pairs_by_relation = parse_raw_dependency_file(input_path, rule_relation_by_id)
    kept_pairs = []
    raw_total = sum(len(v) for v in pairs_by_relation.values())
    for relation in sorted(pairs_by_relation):
        kept_pairs.extend(pairs_by_relation[relation])

    with open(output_path, "w", encoding="utf-8") as f:
        for a, b in kept_pairs:
            f.write(f"{int(a)}\t{int(b)}\n")

    print(f"{os.path.basename(input_path)}: kept {len(kept_pairs)} / {raw_total} dependencies -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Write all same-relation dependency pairs to *_filtered.txt without any filtering."
    )
    parser.add_argument("-d", "--dataset", default="codex-m")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--rule_file", default="")
    parser.add_argument("--synergy_file", default="")
    parser.add_argument("--redundancy_file", default="")
    args = parser.parse_args()

    dataset_dir = os.path.join(args.data_root, args.dataset)
    rules_dir = os.path.join(dataset_dir, "rules")

    rule_file = args.rule_file or os.path.join(rules_dir, "rules-1000")
    synergy_file = args.synergy_file or os.path.join(rules_dir, "synergy.txt")
    redundancy_file = args.redundancy_file or os.path.join(rules_dir, "redundancy.txt")

    relation_ids = read_ids(os.path.join(dataset_dir, "relation_ids.del"))
    rule_relation_by_id = parse_rule_file_metadata(rule_file, relation_ids)

    if os.path.exists(synergy_file):
        write_pairs(
            synergy_file,
            os.path.splitext(synergy_file)[0] + "_filtered.txt",
            rule_relation_by_id,
        )

    if os.path.exists(redundancy_file):
        write_pairs(
            redundancy_file,
            os.path.splitext(redundancy_file)[0] + "_filtered.txt",
            rule_relation_by_id,
        )


if __name__ == "__main__":
    main()
