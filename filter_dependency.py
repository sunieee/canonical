#!/usr/bin/env python3
import argparse
import json
import multiprocessing as mp
import os
import pickle
import re
from collections import defaultdict


WORKER_VALID_ACTIVE_RULE_SETS_BY_RELATION = None
WORKER_TRAIN_ACTIVE_RULE_SETS_BY_RELATION = None
WORKER_MIN_VALID = 0
WORKER_MIN_TRAIN = 0


def _init_filter_worker(valid_active_rule_sets_by_relation, train_active_rule_sets_by_relation, min_valid, min_train):
    global WORKER_VALID_ACTIVE_RULE_SETS_BY_RELATION
    global WORKER_TRAIN_ACTIVE_RULE_SETS_BY_RELATION
    global WORKER_MIN_VALID
    global WORKER_MIN_TRAIN
    WORKER_VALID_ACTIVE_RULE_SETS_BY_RELATION = valid_active_rule_sets_by_relation
    WORKER_TRAIN_ACTIVE_RULE_SETS_BY_RELATION = train_active_rule_sets_by_relation
    WORKER_MIN_VALID = int(min_valid)
    WORKER_MIN_TRAIN = int(min_train)

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


def prefilter_candidates_by_support(active_rule_sets, candidates, min_count):
    min_count = int(min_count)
    if min_count <= 0:
        return list(range(len(candidates)))
    if len(candidates) == 0 or len(active_rule_sets) == 0:
        return []

    adj = defaultdict(list)
    for idx, candidate in enumerate(candidates):
        a, b = int(candidate[0]), int(candidate[1])
        adj[a].append((b, idx))

    counts = [0] * len(candidates)
    keep = [False] * len(candidates)
    remaining = len(candidates)

    for rs in active_rule_sets:
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
                    if counts[idx] >= min_count:
                        keep[idx] = True
                        remaining -= 1

    return [i for i, k in enumerate(keep) if k]


def load_split_targets(split_path):
    split_sp_to_o = defaultdict(list)
    split_po_to_s = defaultdict(list)
    with open(split_path, "r", encoding="utf-8") as f:
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
            split_sp_to_o[(s, p)].append(o)
            split_po_to_s[(p, o)].append(s)
    return dict(split_sp_to_o), dict(split_po_to_s)


def load_applied_rules(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_processed_from_applied(applied_rules, entity_id_to_idx, relation_id_to_idx):
    processed_sp = {}
    processed_po = {}

    tail_applied = applied_rules.get("tail", {})
    for rel_raw, source_map in tail_applied.items():
        if rel_raw not in relation_id_to_idx:
            continue
        p_idx = relation_id_to_idx[rel_raw]
        for s_raw, target_map in source_map.items():
            if s_raw not in entity_id_to_idx:
                continue
            s_idx = entity_id_to_idx[s_raw]
            key = (s_idx, p_idx)
            bucket = processed_sp.setdefault(key, {"candidates": [], "rules": []})

            for o_raw, rule_ids in target_map.items():
                if o_raw not in entity_id_to_idx:
                    continue
                o_idx = entity_id_to_idx[o_raw]
                ids = [int(rid) for rid in rule_ids if int(rid) > 0]
                bucket["candidates"].append(o_idx)
                bucket["rules"].append(ids)

    head_applied = applied_rules.get("head", {})
    for rel_raw, source_map in head_applied.items():
        if rel_raw not in relation_id_to_idx:
            continue
        p_idx = relation_id_to_idx[rel_raw]
        for o_raw, target_map in source_map.items():
            if o_raw not in entity_id_to_idx:
                continue
            o_idx = entity_id_to_idx[o_raw]
            key = (p_idx, o_idx)
            bucket = processed_po.setdefault(key, {"candidates": [], "rules": []})

            for s_raw, rule_ids in target_map.items():
                if s_raw not in entity_id_to_idx:
                    continue
                s_idx = entity_id_to_idx[s_raw]
                ids = [int(rid) for rid in rule_ids if int(rid) > 0]
                bucket["candidates"].append(s_idx)
                bucket["rules"].append(ids)

    return processed_sp, processed_po


def load_processed_train(expl_dir, entity_ids, relation_ids):
    sp_path = os.path.join(expl_dir, "processed_sp_train.pkl")
    po_path = os.path.join(expl_dir, "processed_po_train.pkl")
    if os.path.exists(sp_path) and os.path.exists(po_path):
        return pickle.load(open(sp_path, "rb")), pickle.load(open(po_path, "rb"))

    applied_path = os.path.join(expl_dir, "applied_rules_train.json")
    if not os.path.exists(applied_path):
        raise FileNotFoundError(
            f"Missing processed train explanations ({sp_path}, {po_path}) and fallback source {applied_path}"
        )

    entity_id_to_idx = {ent: idx for idx, ent in enumerate(entity_ids)}
    relation_id_to_idx = {rel: idx for idx, rel in enumerate(relation_ids)}
    applied_rules_train = load_applied_rules(applied_path)
    return build_processed_from_applied(applied_rules_train, entity_id_to_idx, relation_id_to_idx)


def build_active_rule_sets_by_relation(split_to_targets, processed_sp, processed_po):
    active_rule_sets_by_relation = defaultdict(list)
    for relation, sets_ in collect_positive_active_rule_sets_by_relation(split_to_targets[0], processed_sp, "o").items():
        active_rule_sets_by_relation[int(relation)].extend(sets_)
    for relation, sets_ in collect_positive_active_rule_sets_by_relation(split_to_targets[1], processed_po, "s").items():
        active_rule_sets_by_relation[int(relation)].extend(sets_)
    return active_rule_sets_by_relation


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


def filter_relation_candidates(task):
    relation, chunk_id, candidates = task
    keep_idx = set(range(len(candidates)))

    if WORKER_MIN_VALID > 0:
        active_valid = WORKER_VALID_ACTIVE_RULE_SETS_BY_RELATION.get(int(relation), [])
        keep_idx &= set(prefilter_candidates_by_support(active_valid, candidates, WORKER_MIN_VALID))

    if WORKER_MIN_TRAIN > 0:
        active_train = WORKER_TRAIN_ACTIVE_RULE_SETS_BY_RELATION.get(int(relation), [])
        keep_idx &= set(prefilter_candidates_by_support(active_train, candidates, WORKER_MIN_TRAIN))

    keep_idx = sorted(keep_idx)
    kept_pairs = [candidates[i] for i in keep_idx]
    return int(relation), int(chunk_id), int(len(candidates)), kept_pairs


def build_filter_tasks(pairs_by_relation, jobs, chunk_candidates):
    relation_items = sorted(pairs_by_relation.items(), key=lambda x: x[0])
    tasks = []
    for relation, candidates in relation_items:
        candidate_count = len(candidates)
        if candidate_count == 0:
            continue

        # Small relations stay as a single task to avoid extra overhead.
        if jobs <= 1 or candidate_count <= chunk_candidates:
            tasks.append((int(relation), 0, candidates))
            continue

        for chunk_id, start in enumerate(range(0, candidate_count, chunk_candidates)):
            end = min(start + chunk_candidates, candidate_count)
            tasks.append((int(relation), int(chunk_id), candidates[start:end]))
    return tasks


def filter_dependency_file(
    input_path,
    output_path,
    valid_active_rule_sets_by_relation,
    train_active_rule_sets_by_relation,
    rule_relation_by_id,
    min_valid,
    min_train,
    min_abs_lift,
    jobs=1,
    progress_every=10,
    chunk_candidates=50000,
):
    pairs_by_relation = parse_raw_dependency_file(input_path, rule_relation_by_id, min_abs_lift=min_abs_lift)

    kept_pairs = []
    raw_total = sum(len(v) for v in pairs_by_relation.values())
    relation_items = sorted(pairs_by_relation.items(), key=lambda x: x[0])
    num_relations = len(relation_items)
    jobs = max(int(jobs), 1)
    progress_every = max(int(progress_every), 1)
    chunk_candidates = max(int(chunk_candidates), 1)
    tasks = build_filter_tasks(pairs_by_relation, jobs=jobs, chunk_candidates=chunk_candidates)
    num_tasks = len(tasks)

    print(
        f"{os.path.basename(input_path)}: filtering {raw_total} dependencies across {num_relations} relations "
        f"using {num_tasks} task(s) (jobs={jobs}, chunk_candidates={chunk_candidates}, "
        f"min_valid={int(min_valid)}, min_train={int(min_train)}, min_abs_lift={float(min_abs_lift)})"
    )

    processed_tasks = 0
    processed_candidates = 0

    if jobs == 1:
        _init_filter_worker(
            valid_active_rule_sets_by_relation,
            train_active_rule_sets_by_relation,
            min_valid,
            min_train,
        )
        for item in tasks:
            _relation, _chunk_id, candidate_count, kept_pairs_rel = filter_relation_candidates(item)
            kept_pairs.extend(kept_pairs_rel)
            processed_tasks += 1
            processed_candidates += candidate_count
            if processed_tasks % progress_every == 0 or processed_tasks == num_tasks:
                print(
                    f"{os.path.basename(input_path)}: progress {processed_tasks}/{num_tasks} tasks, "
                    f"{processed_candidates}/{raw_total} candidates examined, kept={len(kept_pairs)}"
                )
    else:
        with mp.Pool(
            processes=jobs,
            initializer=_init_filter_worker,
            initargs=(
                valid_active_rule_sets_by_relation,
                train_active_rule_sets_by_relation,
                min_valid,
                min_train,
            ),
        ) as pool:
            for _relation, _chunk_id, candidate_count, kept_pairs_rel in pool.imap_unordered(
                filter_relation_candidates, tasks, chunksize=1
            ):
                kept_pairs.extend(kept_pairs_rel)
                processed_tasks += 1
                processed_candidates += candidate_count
                if processed_tasks % progress_every == 0 or processed_tasks == num_tasks:
                    print(
                        f"{os.path.basename(input_path)}: progress {processed_tasks}/{num_tasks} tasks, "
                        f"{processed_candidates}/{raw_total} candidates examined, kept={len(kept_pairs)}"
                    )

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
    parser.add_argument("--min_valid", type=int, default=3)
    parser.add_argument("--min_train", type=int, default=0)
    parser.add_argument("--min_abs_lift", type=float, default=0.01)
    parser.add_argument("--jobs", type=int, default=1, help="Number of worker processes for relation-level filtering.")
    parser.add_argument(
        "--progress_every",
        type=int,
        default=10,
        help="Print progress after every N filtering tasks per dependency file.",
    )
    parser.add_argument(
        "--chunk_candidates",
        type=int,
        default=50000,
        help="Split very large relations into chunks of this many candidate pairs before dispatching work.",
    )
    args = parser.parse_args()

    dataset_dir = os.path.join(args.data_root, args.dataset)
    rules_dir = os.path.join(dataset_dir, "rules")
    expl_dir = os.path.join(dataset_dir, "expl")

    rule_file = args.rule_file or os.path.join(rules_dir, "rules-1000")
    synergy_file = args.synergy_file or os.path.join(rules_dir, "synergy.txt")
    redundancy_file = args.redundancy_file or os.path.join(rules_dir, "redundancy.txt")

    relation_ids = read_ids(os.path.join(dataset_dir, "relation_ids.del"))
    rule_relation_by_id = parse_rule_file_metadata(rule_file, relation_ids)

    valid_active_rule_sets_by_relation = defaultdict(list)
    if int(args.min_valid) > 0:
        valid_sp_to_o, valid_po_to_s = load_split_targets(os.path.join(dataset_dir, "valid.del"))
        processed_sp_valid = pickle.load(open(os.path.join(expl_dir, "processed_sp_valid.pkl"), "rb"))
        processed_po_valid = pickle.load(open(os.path.join(expl_dir, "processed_po_valid.pkl"), "rb"))
        valid_active_rule_sets_by_relation = build_active_rule_sets_by_relation(
            (valid_sp_to_o, valid_po_to_s),
            processed_sp_valid,
            processed_po_valid,
        )

    train_active_rule_sets_by_relation = defaultdict(list)
    if int(args.min_train) > 0:
        entity_ids = read_ids(os.path.join(dataset_dir, "entity_ids.del"))
        train_sp_to_o, train_po_to_s = load_split_targets(os.path.join(dataset_dir, "train.del"))
        processed_sp_train, processed_po_train = load_processed_train(expl_dir, entity_ids, relation_ids)
        train_active_rule_sets_by_relation = build_active_rule_sets_by_relation(
            (train_sp_to_o, train_po_to_s),
            processed_sp_train,
            processed_po_train,
        )

    if os.path.exists(synergy_file):
        filter_dependency_file(
            synergy_file,
            os.path.splitext(synergy_file)[0] + "_filtered.txt",
            valid_active_rule_sets_by_relation,
            train_active_rule_sets_by_relation,
            rule_relation_by_id,
            min_valid=args.min_valid,
            min_train=args.min_train,
            min_abs_lift=args.min_abs_lift,
            jobs=args.jobs,
            progress_every=args.progress_every,
            chunk_candidates=args.chunk_candidates,
        )

    if os.path.exists(redundancy_file):
        filter_dependency_file(
            redundancy_file,
            os.path.splitext(redundancy_file)[0] + "_filtered.txt",
            valid_active_rule_sets_by_relation,
            train_active_rule_sets_by_relation,
            rule_relation_by_id,
            min_valid=args.min_valid,
            min_train=args.min_train,
            min_abs_lift=args.min_abs_lift,
            jobs=args.jobs,
            progress_every=args.progress_every,
            chunk_candidates=args.chunk_candidates,
        )


if __name__ == "__main__":
    main()
