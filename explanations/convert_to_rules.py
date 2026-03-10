import argparse
import ast
import json
import os
import re
from collections import defaultdict, deque
from typing import Dict, List, Tuple


def _extract_head_relation(rule_str: str) -> str:
    head = rule_str.split(" <= ", 1)[0].strip()
    if "(" not in head:
        return ""
    return head.split("(", 1)[0].strip()


def _split_rule_line(line: str) -> List[str]:
    line = line.rstrip("\n")
    parts = line.split("\t")
    if len(parts) >= 4:
        return parts
    return re.split(r"\s+", line.strip(), maxsplit=3)


def parse_rules_file(rules_file: str):
    relation_entries = defaultdict(list)
    with open(rules_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            parts = _split_rule_line(line)
            if len(parts) < 4:
                continue
            num_preds = parts[0].strip()
            num_true = parts[1].strip()
            rule_str = parts[3].strip()
            rel = _extract_head_relation(rule_str)
            if not rel:
                continue
            relation_entries[rel].append(
                {
                    "line_no": int(line_no),
                    "sig_full": (num_preds, num_true, rule_str),
                    "sig_rule": rule_str,
                }
            )
    return relation_entries


def parse_rules_index_file(rules_index_file: str):
    relation_entries = defaultdict(list)
    with open(rules_index_file, "r", encoding="utf-8") as f:
        text = f.read()

    blocks = text.split(">>>")[1:]
    for block in blocks:
        raw = block.split("\n")
        rel = raw[0].strip()
        if rel == "":
            continue

        rules_raw = raw[1:]
        for rule in rules_raw:
            if rule.strip() == "":
                continue
            parts = rule.split("\t")
            if len(parts) < 5:
                continue
            local_idx = int(parts[0])
            num_preds = parts[1].strip()
            num_true = parts[2].strip()
            rule_str = parts[4].strip()
            relation_entries[rel].append(
                {
                    "local_idx": local_idx,
                    "sig_full": (num_preds, num_true, rule_str),
                    "sig_rule": rule_str,
                }
            )

    for rel in relation_entries:
        relation_entries[rel].sort(key=lambda x: x["local_idx"])
    return relation_entries


def build_local_to_global_map(rules_entries, rules_index_entries):
    local_to_global = {}

    for rel, idx_rules in rules_index_entries.items():
        rel_rules = rules_entries.get(rel, [])
        used = set()

        by_full = defaultdict(deque)
        by_rule = defaultdict(deque)
        for entry in rel_rules:
            line_no = int(entry["line_no"])
            by_full[entry["sig_full"]].append(line_no)
            by_rule[entry["sig_rule"]].append(line_no)

        rel_map = {}
        for idx_entry in idx_rules:
            local_idx = int(idx_entry["local_idx"])
            mapped = None

            queue_full = by_full.get(idx_entry["sig_full"], deque())
            while queue_full and queue_full[0] in used:
                queue_full.popleft()
            if queue_full:
                mapped = int(queue_full.popleft())

            if mapped is None:
                queue_rule = by_rule.get(idx_entry["sig_rule"], deque())
                while queue_rule and queue_rule[0] in used:
                    queue_rule.popleft()
                if queue_rule:
                    mapped = int(queue_rule.popleft())

            if mapped is None and local_idx < len(rel_rules):
                candidate = int(rel_rules[local_idx]["line_no"])
                if candidate not in used:
                    mapped = candidate

            if mapped is None:
                for entry in rel_rules:
                    candidate = int(entry["line_no"])
                    if candidate not in used:
                        mapped = candidate
                        break

            if mapped is None:
                raise RuntimeError(
                    f"Cannot map local rule index {local_idx} for relation '{rel}' to rules file line number"
                )

            used.add(mapped)
            rel_map[local_idx] = mapped

        local_to_global[rel] = rel_map

    return local_to_global


def parse_explanations(explanation_file: str):
    with open(explanation_file, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = text.split("}}},")
    if chunks and chunks[-1].strip() == "":
        chunks = chunks[:-1]

    for chunk in chunks:
        proc = chunk.replace("\n", "") + "}}}"
        raw_dict = ast.literal_eval(proc)
        triple = list(raw_dict.keys())[0]
        meta = list(raw_dict.values())[0]
        yield triple, meta


def convert_to_applied_rules(explanation_file: str, local_to_global_map):
    applied = {"head": {}, "tail": {}}

    for triple, meta in parse_explanations(explanation_file):
        s, p, o = triple.split(" ")

        heads = meta["heads"]
        tails = meta["tails"]

        head_candidates = list(heads["candidates"])
        tail_candidates = list(tails["candidates"])

        if "me_myself_i" in head_candidates:
            idx = head_candidates.index("me_myself_i")
            head_candidates[idx] = o
        if "me_myself_i" in tail_candidates:
            idx = tail_candidates.index("me_myself_i")
            tail_candidates[idx] = s

        rel_local_map = local_to_global_map.get(p, {})

        head_bucket = applied["head"].setdefault(p, {}).setdefault(o, {})
        for cand, local_ids in zip(head_candidates, heads["rules"]):
            mapped = [int(rel_local_map[rid]) for rid in local_ids if rid in rel_local_map]
            head_bucket[cand] = mapped

        tail_bucket = applied["tail"].setdefault(p, {}).setdefault(s, {})
        for cand, local_ids in zip(tail_candidates, tails["rules"]):
            mapped = [int(rel_local_map[rid]) for rid in local_ids if rid in rel_local_map]
            tail_bucket[cand] = mapped

    return applied


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--explanation_file", required=True)
    parser.add_argument("--rules_index_file", required=True)
    parser.add_argument("--rules_file", required=True)
    parser.add_argument("--output_file", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    rules_entries = parse_rules_file(args.rules_file)
    rules_index_entries = parse_rules_index_file(args.rules_index_file)
    local_to_global_map = build_local_to_global_map(rules_entries, rules_index_entries)

    applied = convert_to_applied_rules(args.explanation_file, local_to_global_map)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(applied, f, ensure_ascii=False)

    print(f"Saved applied rules to {args.output_file}")


if __name__ == "__main__":
    main()
