#!/usr/bin/env python3
import argparse
import math
import os
import pickle
import re
from collections import defaultdict
from itertools import combinations
from multiprocessing import cpu_count, get_context

try:
    from tqdm import tqdm
except Exception:
    class tqdm:  # type: ignore
        def __init__(self, iterable=None, total=None, desc="", unit="it"):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.unit = unit
            self.n = 0
            self._last_print_n = -1
            if self.iterable is None:
                prefix = f"{self.desc}: " if self.desc else ""
                if self.total is not None:
                    print(f"{prefix}0/{self.total} {self.unit}")
                else:
                    print(f"{prefix}started")

        def _should_print(self):
            if self.total is None:
                return True
            step = max(int(self.total // 100), 1000)
            return self.n == self.total or self.n - self._last_print_n >= step

        def _print(self):
            self._last_print_n = self.n
            print(f"{self.desc}: {self.n}/{self.total} {self.unit}", end="\r", flush=True)

        def update(self, n=1):
            self.n += n
            if self.total is not None and self._should_print():
                self._print()

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            for item in self.iterable:
                self.n += 1
                if self.total is not None and self._should_print():
                    self._print()
                yield item
            if self.total is not None:
                print(f"{self.desc}: {self.n}/{self.total} {self.unit}")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            if self.total is not None:
                print(f"{self.desc}: {self.n}/{self.total} {self.unit}")


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
    rule_conf_by_id = {}

    with open(rule_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            parts = _split_rule_line(line)
            if len(parts) < 4:
                continue
            rel = extract_head_relation(parts[3].strip())
            rel_id = relation_to_id.get(rel)
            if rel_id is None:
                continue
            try:
                conf = float(parts[2])
            except Exception:
                conf = 0.0
            conf = min(max(conf, 0.0), 1.0 - 1e-7)
            rule_relation_by_id[int(line_no)] = int(rel_id)
            rule_conf_by_id[int(line_no)] = float(conf)
    return rule_relation_by_id, rule_conf_by_id


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
    by_relation = defaultdict(list)
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
                body_size = int(float(parts[0]))
                support = int(float(parts[1]))
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
            by_relation[int(rel1)].append(
                {
                    "relation": int(rel1),
                    "a": int(a),
                    "b": int(b),
                    "body_size": int(body_size),
                    "support_file": int(support),
                    "lift": float(lift),
                }
            )

    return dict(by_relation)


def build_rule_conf_table(rule_conf_by_id):
    if len(rule_conf_by_id) == 0:
        return [0.0]
    max_rule_id = max(int(rid) for rid in rule_conf_by_id.keys())
    conf_table = [0.0] * (max_rule_id + 1)
    for rid, conf in rule_conf_by_id.items():
        conf_table[int(rid)] = float(conf)
    return conf_table


def compute_base_probability(rule_ids, rule_conf_table, mode):
    confs = [rule_conf_table[int(rid)] if int(rid) < len(rule_conf_table) else 0.0 for rid in rule_ids]
    if len(confs) == 0:
        return 0.0

    if mode == "max":
        return max(confs)
    if mode == "sum":
        return min(max(sum(confs), 0.0), 1.0 - 1e-7)

    # Default: noisy-or, aligned with independent evidence accumulation.
    log_prod = 0.0
    for conf in confs:
        conf = min(max(float(conf), 0.0), 1.0 - 1e-7)
        log_prod += math.log1p(-conf)
    p = 1.0 - math.exp(log_prod)
    return min(max(p, 1e-7), 1.0 - 1e-7)


def iter_valid_samples(valid_targets, processed, direction, rule_conf_by_id, base_score_mode):
    for key, golds in valid_targets.items():
        if direction == "o":
            _e, relation = key
        else:
            relation, _e = key

        bucket = processed.get(key)
        if bucket is None:
            continue

        if hasattr(golds, "tolist"):
            gold_iter = golds.tolist()
        else:
            gold_iter = golds
        gold_set = set(int(x) for x in gold_iter)

        candidates = bucket.get("candidates", [])
        rules_per_candidate = bucket.get("rules", [])
        for prediction, rule_ids in zip(candidates, rules_per_candidate):
            unique_rules = sorted(set(int(rid) for rid in rule_ids if int(rid) > 0))
            if len(unique_rules) == 0:
                continue
            y = 1.0 if int(prediction) in gold_set else 0.0
            p_base = compute_base_probability(unique_rules, rule_conf_by_id, base_score_mode)
            residual = y - p_base
            yield int(relation), unique_rules, float(y), float(p_base), float(residual)


def build_relation_candidate_index(synergy_by_relation, redundancy_by_relation):
    relation_info = {}
    all_relations = sorted(set(synergy_by_relation.keys()) | set(redundancy_by_relation.keys()))
    for relation in all_relations:
        synergy_candidates = list(synergy_by_relation.get(relation, []))
        redundancy_candidates = list(redundancy_by_relation.get(relation, []))
        pair_set = set()
        adj = defaultdict(list)
        candidate_rule_set = set()
        for cand in synergy_candidates + redundancy_candidates:
            key = (int(cand["a"]), int(cand["b"]))
            if key in pair_set:
                continue
            pair_set.add(key)
            adj[key[0]].append((key[1], key))
            candidate_rule_set.add(key[0])
            candidate_rule_set.add(key[1])
        degree = {int(a): len(bs) for a, bs in adj.items()}
        relation_info[int(relation)] = {
            "synergy_candidates": synergy_candidates,
            "redundancy_candidates": redundancy_candidates,
            "pair_set": pair_set,
            "adj": dict(adj),
            "degree": degree,
            "candidate_rule_set": candidate_rule_set,
        }
    return relation_info


def count_total_candidates(processed):
    total = 0
    for bucket in processed.values():
        total += len(bucket.get("candidates", []))
    return int(total)


def accumulate_residual_stats_stream(
    valid_targets,
    processed,
    direction,
    rule_conf_table,
    base_score_mode,
    relation_info,
    single_sum,
    single_count,
    pair_sum,
    pair_count,
    progress,
):
    base_prob_cache = {}
    cache_limit = 200000

    for key, golds in valid_targets.items():
        if direction == "o":
            _e, relation = key
        else:
            relation, _e = key

        bucket = processed.get(key)
        if bucket is None:
            continue

        rel_info = relation_info.get(int(relation))
        candidates = bucket.get("candidates", [])
        rules_per_candidate = bucket.get("rules", [])
        if hasattr(golds, "tolist"):
            gold_iter = golds.tolist()
        else:
            gold_iter = golds
        gold_set = set(int(x) for x in gold_iter)

        for prediction, rule_ids in zip(candidates, rules_per_candidate):
            if rel_info is None:
                continue

            unique_rules = sorted(set(int(rid) for rid in rule_ids if int(rid) > 0))
            if len(unique_rules) == 0:
                continue

            filtered_rules = [rid for rid in unique_rules if rid in rel_info["candidate_rule_set"]]
            if len(filtered_rules) == 0:
                continue

            filtered_rules_t = tuple(filtered_rules)
            p_base = base_prob_cache.get(filtered_rules_t)
            if p_base is None:
                p_base = compute_base_probability(filtered_rules_t, rule_conf_table, base_score_mode)
                if len(base_prob_cache) >= cache_limit:
                    base_prob_cache.clear()
                base_prob_cache[filtered_rules_t] = p_base

            y = 1.0 if int(prediction) in gold_set else 0.0
            residual = y - p_base

            for rid in filtered_rules_t:
                single_sum[relation][rid] += residual
                single_count[relation][rid] += 1

            if len(filtered_rules_t) < 2:
                continue

            pair_set = rel_info["pair_set"]
            adj = rel_info["adj"]
            degree = rel_info["degree"]
            combo_cost = (len(filtered_rules_t) * (len(filtered_rules_t) - 1)) // 2
            adj_cost = sum(int(degree.get(rid, 0)) for rid in filtered_rules_t)

            if combo_cost <= adj_cost:
                for a, b in combinations(filtered_rules_t, 2):
                    key_ab = (int(a), int(b))
                    if key_ab in pair_set:
                        pair_sum[relation][key_ab] += residual
                        pair_count[relation][key_ab] += 1
            else:
                active = set(filtered_rules_t)
                for a in filtered_rules_t:
                    for b, key_ab in adj.get(a, []):
                        if b in active:
                            pair_sum[relation][key_ab] += residual
                            pair_count[relation][key_ab] += 1

        progress.update(len(candidates))


def compute_interaction_score(
    a,
    b,
    pair_key,
    single_sum,
    single_count,
    pair_sum,
    pair_count,
    support_smoothing,
):
    a_count = int(single_count.get(a, 0))
    b_count = int(single_count.get(b, 0))
    ab_count = int(pair_count.get(pair_key, 0))
    if a_count <= 0 or b_count <= 0 or ab_count <= 0:
        return None

    mean_a = float(single_sum[a] / a_count)
    mean_b = float(single_sum[b] / b_count)
    mean_ab = float(pair_sum[pair_key] / ab_count)

    raw_interaction = mean_ab - ((mean_a + mean_b) / 2.0)
    shrink = float(ab_count / (ab_count + max(float(support_smoothing), 1.0)))
    shrunk_interaction = raw_interaction * shrink

    return {
        "rule_a_mean_residual": mean_a,
        "rule_b_mean_residual": mean_b,
        "pair_mean_residual": mean_ab,
        "valid_pair_support": ab_count,
        "interaction_raw": raw_interaction,
        "interaction_score": shrunk_interaction,
    }


def score_candidates_for_relation(
    relation,
    candidates,
    pair_type,
    single_sum,
    single_count,
    pair_sum,
    pair_count,
    min_valid,
    min_abs_score,
    support_smoothing,
    top_k_per_relation,
):
    detail_rows = []
    scored = []

    for cand in candidates:
        a = int(cand["a"])
        b = int(cand["b"])
        pair_key = (a, b)
        score_info = compute_interaction_score(
            a=a,
            b=b,
            pair_key=pair_key,
            single_sum=single_sum,
            single_count=single_count,
            pair_sum=pair_sum,
            pair_count=pair_count,
            support_smoothing=support_smoothing,
        )
        if score_info is None:
            continue

        keep = False
        final_score = float(score_info["interaction_score"])
        valid_support = int(score_info["valid_pair_support"])
        if valid_support >= int(min_valid):
            if pair_type == "synergy":
                keep = final_score >= float(min_abs_score)
            else:
                keep = final_score <= -float(min_abs_score)

        row = {
            "relation": int(relation),
            "pair_type": str(pair_type),
            "a": a,
            "b": b,
            "body_size": int(cand["body_size"]),
            "support_file": int(cand["support_file"]),
            "lift": float(cand["lift"]),
            **score_info,
            "keep": bool(keep),
        }
        detail_rows.append(row)
        if keep:
            scored.append(row)

    if pair_type == "synergy":
        scored.sort(key=lambda x: (x["interaction_score"], abs(x["lift"]), x["support_file"]), reverse=True)
    else:
        scored.sort(key=lambda x: (x["interaction_score"], -abs(x["lift"]), -x["support_file"]))

    if int(top_k_per_relation) > 0:
        scored = scored[: int(top_k_per_relation)]

    kept_pairs = [(row["a"], row["b"]) for row in scored]
    return kept_pairs, detail_rows


def _score_relation_task(task):
    relation = int(task["relation"])
    single_sum = task["single_sum"]
    single_count = task["single_count"]
    pair_sum = task["pair_sum"]
    pair_count = task["pair_count"]
    min_valid = task["min_valid"]
    min_abs_score = task["min_abs_score"]
    support_smoothing = task["support_smoothing"]
    top_k_per_relation = task["top_k_per_relation"]

    synergy_kept, synergy_details = score_candidates_for_relation(
        relation=relation,
        candidates=task["synergy_candidates"],
        pair_type="synergy",
        single_sum=single_sum,
        single_count=single_count,
        pair_sum=pair_sum,
        pair_count=pair_count,
        min_valid=min_valid,
        min_abs_score=min_abs_score,
        support_smoothing=support_smoothing,
        top_k_per_relation=top_k_per_relation,
    )
    redundancy_kept, redundancy_details = score_candidates_for_relation(
        relation=relation,
        candidates=task["redundancy_candidates"],
        pair_type="redundancy",
        single_sum=single_sum,
        single_count=single_count,
        pair_sum=pair_sum,
        pair_count=pair_count,
        min_valid=min_valid,
        min_abs_score=min_abs_score,
        support_smoothing=support_smoothing,
        top_k_per_relation=top_k_per_relation,
    )
    return {
        "relation": relation,
        "synergy_kept": synergy_kept,
        "synergy_details": synergy_details,
        "redundancy_kept": redundancy_kept,
        "redundancy_details": redundancy_details,
    }


def write_filtered_pairs(output_path, kept_pairs):
    with open(output_path, "w", encoding="utf-8") as f:
        for a, b in kept_pairs:
            f.write(f"{int(a)}\t{int(b)}\n")


def write_detail_rows(output_path, detail_rows):
    if len(detail_rows) == 0:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(
                "relation\tpair_type\trule1\trule2\tbody_size\tsupport_file\tlift\t"
                "valid_pair_support\trule1_mean_residual\trule2_mean_residual\t"
                "pair_mean_residual\tinteraction_raw\tinteraction_score\tkeep\n"
            )
        return

    detail_rows = sorted(
        detail_rows,
        key=lambda x: (x["relation"], x["pair_type"], -abs(x["interaction_score"]), x["a"], x["b"]),
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(
            "relation\tpair_type\trule1\trule2\tbody_size\tsupport_file\tlift\t"
            "valid_pair_support\trule1_mean_residual\trule2_mean_residual\t"
            "pair_mean_residual\tinteraction_raw\tinteraction_score\tkeep\n"
        )
        for row in detail_rows:
            f.write(
                f"{row['relation']}\t{row['pair_type']}\t{row['a']}\t{row['b']}\t{row['body_size']}\t"
                f"{row['support_file']}\t{row['lift']:.7f}\t{row['valid_pair_support']}\t"
                f"{row['rule_a_mean_residual']:.7f}\t{row['rule_b_mean_residual']:.7f}\t"
                f"{row['pair_mean_residual']:.7f}\t{row['interaction_raw']:.7f}\t"
                f"{row['interaction_score']:.7f}\t{int(row['keep'])}\n"
            )


def run_residual_ranker(
    dataset_dir,
    rule_file,
    synergy_file,
    redundancy_file,
    min_valid,
    min_abs_lift,
    min_abs_score,
    support_smoothing,
    top_k_per_relation,
    base_score_mode,
    num_workers,
):
    expl_dir = os.path.join(dataset_dir, "expl")
    print("Loading valid targets...")
    valid_sp_to_o, valid_po_to_s = load_valid_targets(os.path.join(dataset_dir, "valid.del"))
    print("Loading processed valid explanations...")
    processed_sp_valid = pickle.load(open(os.path.join(expl_dir, "processed_sp_valid.pkl"), "rb"))
    processed_po_valid = pickle.load(open(os.path.join(expl_dir, "processed_po_valid.pkl"), "rb"))

    print("Parsing rule metadata...")
    relation_ids = read_ids(os.path.join(dataset_dir, "relation_ids.del"))
    rule_relation_by_id, rule_conf_by_id = parse_rule_file_metadata(rule_file, relation_ids)
    rule_conf_table = build_rule_conf_table(rule_conf_by_id)

    synergy_by_relation = {}
    redundancy_by_relation = {}
    if os.path.exists(synergy_file):
        print(f"Parsing synergy candidates from {synergy_file}...")
        synergy_by_relation = parse_raw_dependency_file(synergy_file, rule_relation_by_id, min_abs_lift=min_abs_lift)
    if os.path.exists(redundancy_file):
        print(f"Parsing redundancy candidates from {redundancy_file}...")
        redundancy_by_relation = parse_raw_dependency_file(
            redundancy_file, rule_relation_by_id, min_abs_lift=min_abs_lift
        )

    relation_info = build_relation_candidate_index(synergy_by_relation, redundancy_by_relation)
    total_candidates = count_total_candidates(processed_sp_valid) + count_total_candidates(processed_po_valid)
    print(
        f"Candidate relations={len(relation_info)}, "
        f"valid candidate examples={total_candidates}, "
        f"workers={max(int(num_workers), 1)}"
    )

    single_sum = defaultdict(lambda: defaultdict(float))
    single_count = defaultdict(lambda: defaultdict(int))
    pair_sum = defaultdict(lambda: defaultdict(float))
    pair_count = defaultdict(lambda: defaultdict(int))

    with tqdm(total=total_candidates, desc="valid residual pass", unit="cand") as progress:
        accumulate_residual_stats_stream(
            valid_targets=valid_sp_to_o,
            processed=processed_sp_valid,
            direction="o",
            rule_conf_table=rule_conf_table,
            base_score_mode=base_score_mode,
            relation_info=relation_info,
            single_sum=single_sum,
            single_count=single_count,
            pair_sum=pair_sum,
            pair_count=pair_count,
            progress=progress,
        )
        accumulate_residual_stats_stream(
            valid_targets=valid_po_to_s,
            processed=processed_po_valid,
            direction="s",
            rule_conf_table=rule_conf_table,
            base_score_mode=base_score_mode,
            relation_info=relation_info,
            single_sum=single_sum,
            single_count=single_count,
            pair_sum=pair_sum,
            pair_count=pair_count,
            progress=progress,
        )

    tasks = []
    for relation, rel_info in relation_info.items():
        tasks.append(
            {
                "relation": int(relation),
                "single_sum": dict(single_sum.get(relation, {})),
                "single_count": dict(single_count.get(relation, {})),
                "pair_sum": dict(pair_sum.get(relation, {})),
                "pair_count": dict(pair_count.get(relation, {})),
                "synergy_candidates": list(rel_info["synergy_candidates"]),
                "redundancy_candidates": list(rel_info["redundancy_candidates"]),
                "min_valid": int(min_valid),
                "min_abs_score": float(min_abs_score),
                "support_smoothing": float(support_smoothing),
                "top_k_per_relation": int(top_k_per_relation),
            }
        )

    results = []
    score_desc = "score relations"
    if max(int(num_workers), 1) > 1 and len(tasks) > 1:
        with get_context("fork").Pool(processes=max(int(num_workers), 1)) as pool:
            for result in tqdm(pool.imap_unordered(_score_relation_task, tasks), total=len(tasks), desc=score_desc):
                results.append(result)
    else:
        for task in tqdm(tasks, total=len(tasks), desc=score_desc):
            results.append(_score_relation_task(task))

    synergy_kept_pairs = []
    synergy_detail_rows = []
    redundancy_kept_pairs = []
    redundancy_detail_rows = []
    for result in results:
        synergy_kept_pairs.extend(result["synergy_kept"])
        synergy_detail_rows.extend(result["synergy_details"])
        redundancy_kept_pairs.extend(result["redundancy_kept"])
        redundancy_detail_rows.extend(result["redundancy_details"])

    if os.path.exists(synergy_file):
        synergy_output = os.path.splitext(synergy_file)[0] + "_filtered.txt"
        synergy_details = os.path.splitext(synergy_file)[0] + "_filtered_details.tsv"
        write_filtered_pairs(synergy_output, synergy_kept_pairs)
        write_detail_rows(synergy_details, synergy_detail_rows)
        raw_total = sum(len(v) for v in synergy_by_relation.values())
        print(f"synergy: kept {len(synergy_kept_pairs)} / {raw_total} dependencies -> {synergy_output}")
        print(f"synergy details -> {synergy_details}")

    if os.path.exists(redundancy_file):
        redundancy_output = os.path.splitext(redundancy_file)[0] + "_filtered.txt"
        redundancy_details = os.path.splitext(redundancy_file)[0] + "_filtered_details.tsv"
        write_filtered_pairs(redundancy_output, redundancy_kept_pairs)
        write_detail_rows(redundancy_details, redundancy_detail_rows)
        raw_total = sum(len(v) for v in redundancy_by_relation.values())
        print(f"redundancy: kept {len(redundancy_kept_pairs)} / {raw_total} dependencies -> {redundancy_output}")
        print(f"redundancy details -> {redundancy_details}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Filter dependency files by held-out residual interaction on valid candidates. "
            "This is a drop-in replacement for filter_dependency.py that still writes "
            "synergy_filtered.txt / redundancy_filtered.txt."
        )
    )
    parser.add_argument("-d", "--dataset", default="codex-m")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--rule_file", default="")
    parser.add_argument("--synergy_file", default="")
    parser.add_argument("--redundancy_file", default="")
    parser.add_argument("--min_valid", type=int, default=3)
    parser.add_argument("--min_abs_lift", type=float, default=0.01)
    parser.add_argument(
        "--min_abs_score",
        type=float,
        default=0.002,
        help="Minimum absolute shrunk residual interaction score required to keep a pair.",
    )
    parser.add_argument(
        "--support_smoothing",
        type=float,
        default=20.0,
        help="Shrink pair scores toward 0 by count / (count + support_smoothing).",
    )
    parser.add_argument(
        "--top_k_per_relation",
        type=int,
        default=0,
        help="Optional cap on kept pairs per relation. 0 means no cap.",
    )
    parser.add_argument(
        "--base_score_mode",
        choices=["noisy_or", "max", "sum"],
        default="noisy_or",
        help="How to turn active rule confidences into a base probability before computing residuals.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(cpu_count() - 1, 1),
        help="Number of worker processes used for relation-level scoring after the streamed residual pass.",
    )
    args = parser.parse_args()

    dataset_dir = os.path.join(args.data_root, args.dataset)
    rules_dir = os.path.join(dataset_dir, "rules")
    rule_file = args.rule_file or os.path.join(rules_dir, "rules-1000")
    synergy_file = args.synergy_file or os.path.join(rules_dir, "synergy.txt")
    redundancy_file = args.redundancy_file or os.path.join(rules_dir, "redundancy.txt")

    run_residual_ranker(
        dataset_dir=dataset_dir,
        rule_file=rule_file,
        synergy_file=synergy_file,
        redundancy_file=redundancy_file,
        min_valid=args.min_valid,
        min_abs_lift=args.min_abs_lift,
        min_abs_score=args.min_abs_score,
        support_smoothing=args.support_smoothing,
        top_k_per_relation=args.top_k_per_relation,
        base_score_mode=args.base_score_mode,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
