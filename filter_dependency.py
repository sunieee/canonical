#!/usr/bin/env python3
import argparse
import os
import pickle
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


def collect_positive_active_rule_sets_by_relation(split_to_targets, processed, direction="o"):
    active_rule_sets_by_relation = defaultdict(list)
    for key, golds in split_to_targets.items():
        if direction == "o":
            _e, relation = key
        else:
            relation, _e = key

        if key not in processed:
            continue

        if hasattr(golds, "tolist"):
            gold_iter = golds.tolist()
        else:
            gold_iter = golds
        gold_set = set(int(x) for x in gold_iter)
        candidates = processed[key].get("candidates", [])
        rules_per_candidate = processed[key].get("rules", [])
        for prediction, rule_ids in zip(candidates, rules_per_candidate):
            if int(prediction) not in gold_set:
                continue
            active_rule_sets_by_relation[int(relation)].append(set(int(rid) for rid in rule_ids))
    return active_rule_sets_by_relation


def prefilter_candidates_by_valid(active_valid, candidates, min_valid):
    min_valid = int(min_valid)
    if min_valid <= 0:
        return list(range(len(candidates)))
    if len(candidates) == 0 or len(active_valid) == 0:
        return []

    adj = defaultdict(list)
    for idx, candidate in enumerate(candidates):
        a, b = int(candidate[0]), int(candidate[1])
        adj[a].append((b, idx))

    counts = [0] * len(candidates)
    keep = [False] * len(candidates)
    remaining = len(candidates)

    for rs in active_valid:
        if remaining <= 0:
            break
        if len(rs) < 2:
            continue
        for a in rs:
            if a not in adj:
                continue
            for b, idx in adj[a]:
                if keep[idx]:
                    continue
                if b in rs:
                    counts[idx] += 1
                    if counts[idx] >= min_valid:
                        keep[idx] = True
                        remaining -= 1

    return [i for i, k in enumerate(keep) if k]


def load_valid_targets(valid_path):
    valid_sp_to_o = defaultdict(list)
    valid_po_to_s = defaultdict(list)
    with open(valid_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                parts = re.split(r"\s+", line)
            if len(parts) < 3:
                continue
            s, p, o = int(parts[0]), int(parts[1]), int(parts[2])
            valid_sp_to_o[(s, p)].append(o)
            valid_po_to_s[(p, o)].append(s)
    return dict(valid_sp_to_o), dict(valid_po_to_s)


def parse_raw_dependency_file(path, rule_relation_by_id, min_abs_lift):
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
                lift = float(parts[2])
                id1 = int(parts[3])
                id2 = int(parts[4])
            except Exception:
                continue

            if abs(lift) < float(min_abs_lift):
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


def filter_dependency_file(input_path, output_path, valid_active_rule_sets_by_relation, rule_relation_by_id, min_valid, min_abs_lift):
    pairs_by_relation = parse_raw_dependency_file(input_path, rule_relation_by_id, min_abs_lift=min_abs_lift)

    kept_pairs = []
    raw_total = sum(len(v) for v in pairs_by_relation.values())
    for relation, candidates in pairs_by_relation.items():
        active_valid = valid_active_rule_sets_by_relation.get(int(relation), [])
        keep_idx = prefilter_candidates_by_valid(active_valid, candidates, min_valid)
        kept_pairs.extend(candidates[i] for i in keep_idx)

    with open(output_path, "w", encoding="utf-8") as f:
        for a, b in kept_pairs:
            f.write(f"{int(a)}\t{int(b)}\n")

    print(f"{os.path.basename(input_path)}: kept {len(kept_pairs)} / {raw_total} dependencies -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Filter dependency files by valid support and lift magnitude.")
    parser.add_argument("-d", "--dataset", default="codex-m")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--rule_file", default="")
    parser.add_argument("--synergy_file", default="")
    parser.add_argument("--redundancy_file", default="")
    parser.add_argument("--min_valid", type=int, default=5)
    parser.add_argument("--min_abs_lift", type=float, default=0.01)
    args = parser.parse_args()

    dataset_dir = os.path.join(args.data_root, args.dataset)
    rules_dir = os.path.join(dataset_dir, "rules")
    expl_dir = os.path.join(dataset_dir, "expl")

    rule_file = args.rule_file or os.path.join(rules_dir, "rules-1000")
    synergy_file = args.synergy_file or os.path.join(rules_dir, "synergy.txt")
    redundancy_file = args.redundancy_file or os.path.join(rules_dir, "redundancy.txt")

    valid_sp_to_o, valid_po_to_s = load_valid_targets(os.path.join(dataset_dir, "valid.del"))
    processed_sp_valid = pickle.load(open(os.path.join(expl_dir, "processed_sp_valid.pkl"), "rb"))
    processed_po_valid = pickle.load(open(os.path.join(expl_dir, "processed_po_valid.pkl"), "rb"))

    valid_active_rule_sets_by_relation = defaultdict(list)
    for relation, sets_ in collect_positive_active_rule_sets_by_relation(valid_sp_to_o, processed_sp_valid, "o").items():
        valid_active_rule_sets_by_relation[int(relation)].extend(sets_)
    for relation, sets_ in collect_positive_active_rule_sets_by_relation(valid_po_to_s, processed_po_valid, "s").items():
        valid_active_rule_sets_by_relation[int(relation)].extend(sets_)

    relation_ids = read_ids(os.path.join(dataset_dir, "relation_ids.del"))
    rule_relation_by_id = parse_rule_file_metadata(rule_file, relation_ids)

    if os.path.exists(synergy_file):
        filter_dependency_file(
            synergy_file,
            os.path.splitext(synergy_file)[0] + "_filtered.txt",
            valid_active_rule_sets_by_relation,
            rule_relation_by_id,
            min_valid=args.min_valid,
            min_abs_lift=args.min_abs_lift,
        )

    if os.path.exists(redundancy_file):
        filter_dependency_file(
            redundancy_file,
            os.path.splitext(redundancy_file)[0] + "_filtered.txt",
            valid_active_rule_sets_by_relation,
            rule_relation_by_id,
            min_valid=args.min_valid,
            min_abs_lift=args.min_abs_lift,
        )


if __name__ == "__main__":
    main()
