import argparse
import json
import os

from c_clause import Loader, RankingHandler
from clause import Options


def sanitize_applied_rules(applied_rules):
    """
    Ensure applied-rules payload is JSON-safe and keeps only valid 1-based rule ids.
    Expected structure:
      {relation: {source: {target: [rule_id, ...]}}}
    """
    cleaned = {}
    for rel, source_map in applied_rules.items():
        rel_bucket = cleaned.setdefault(rel, {})
        for source, target_map in source_map.items():
            source_bucket = rel_bucket.setdefault(source, {})
            for target, rule_ids in target_map.items():
                ids = []
                for rid in rule_ids:
                    rid = int(rid)
                    if rid > 0:
                        ids.append(rid)
                source_bucket[target] = ids
    return cleaned


def extract_topk_candidates_from_ranking(ranking, topk):
    """
    Build lookup set from ranking output.
    Expected ranking structure:
      {relation: {query_entity: [(candidate, score), ...]}}
    Returns:
      {relation: {query_entity: {candidate, ...}}}
    """
    k = int(topk)
    topk_lookup = {}
    for rel, query_map in ranking.items():
        rel_bucket = topk_lookup.setdefault(rel, {})
        for query, scored_candidates in query_map.items():
            cands = set()
            for item in scored_candidates[:k]:
                if not item:
                    continue
                cand = item[0]
                cands.add(cand)
            rel_bucket[query] = cands
    return topk_lookup


def filter_applied_rules_by_topk(applied_rules, topk_lookup):
    """
    Keep only candidates that appear in ranking top-k for each (relation, query).
    """
    filtered = {}
    for rel, source_map in applied_rules.items():
        rel_topk = topk_lookup.get(rel, {})
        rel_bucket = {}
        for source, target_map in source_map.items():
            allowed_candidates = rel_topk.get(source)
            if allowed_candidates is None:
                continue

            kept_targets = {}
            for target, rule_ids in target_map.items():
                if target in allowed_candidates:
                    kept_targets[target] = rule_ids

            if kept_targets:
                rel_bucket[source] = kept_targets

        if rel_bucket:
            filtered[rel] = rel_bucket
    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="AnyBURL Apply-like output using PyClause ranking + rule features"
    )
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--rules", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--worker-threads", type=int, default=-1)
    parser.add_argument("--aggregation", default="maxplus")
    parser.add_argument("--filter-w-data", type=int, default=1)
    parser.add_argument("--min-correct-predictions", type=int, default=5)
    parser.add_argument("--read-cyclic-rules", type=int, default=1)
    parser.add_argument("--read-acyclic1-rules", type=int, default=1)
    parser.add_argument("--read-acyclic2-rules", type=int, default=0)
    parser.add_argument("--read-zero-rules", type=int, default=0)
    parser.add_argument("--read-uxxc-rules", type=int, default=1)
    parser.add_argument("--read-uxxd-rules", type=int, default=1)

    args = parser.parse_args()
    opts = Options()
    # loader rule selection
    opts.set("loader.load_b_rules", bool(args.read_cyclic_rules))
    opts.set("loader.load_u_c_rules", bool(args.read_acyclic1_rules))
    opts.set("loader.load_u_d_rules", bool(args.read_acyclic2_rules))
    opts.set("loader.load_zero_rules", bool(args.read_zero_rules))
    opts.set("loader.load_u_xxc_rules", bool(args.read_uxxc_rules))
    opts.set("loader.load_u_xxd_rules", bool(args.read_uxxd_rules))
    opts.set("loader.b_min_support", int(args.min_correct_predictions))
    opts.set("loader.c_min_support", int(args.min_correct_predictions))
    # IMPORTANT: default is 1000, which prunes B-rule DFS branches for efficiency.
    # Set to -1 to disable pruning and get exhaustive rule application (matching manual scan).
    opts.set("loader.b_max_branching_factor", -1)
    if args.worker_threads is not None and int(args.worker_threads) > 0:
        opts.set("loader.num_threads", int(args.worker_threads))

    # ranking configuration
    opts.set("ranking_handler.collect_rules", True)
    opts.set("ranking_handler.topk", args.topk)
    opts.set("ranking_handler.aggregation_function", args.aggregation)
    opts.set("ranking_handler.filter_w_data", bool(args.filter_w_data))
    # opts.set("ranking_handler.num_top_rules", -1)
    opts.set("ranking_handler.num_top_rules", 200)
    opts.set("ranking_handler.num_threads", args.worker_threads)
    # make sure we do not stop early
    opts.set("ranking_handler.disc_at_least", -1)

    loader = Loader(options=opts.get("loader"))
    if args.valid.endswith("empty.txt"):
        loader.load_data(data=args.train, target=args.target)
    else:   
        loader.load_data(data=args.train, filter=args.valid, target=args.target)
    loader.load_rules(rules=args.rules)

    ranker = RankingHandler(options=opts.get("ranking_handler"))
    ranker.calculate_ranking(loader=loader)

    # ranking used as authoritative top-k candidate list per query
    head_ranking = ranker.get_ranking(direction="head", as_string=True)
    tail_ranking = ranker.get_ranking(direction="tail", as_string=True)

    # applied rules from PyClause (rule ids align with rules file line numbers, 1-based)
    head_applied_rules = ranker.get_applied_rules(direction="head")
    tail_applied_rules = ranker.get_applied_rules(direction="tail")

    head_topk = extract_topk_candidates_from_ranking(head_ranking, args.topk)
    tail_topk = extract_topk_candidates_from_ranking(tail_ranking, args.topk)

    head_applied_rules = filter_applied_rules_by_topk(head_applied_rules, head_topk)
    tail_applied_rules = filter_applied_rules_by_topk(tail_applied_rules, tail_topk)

    applied_payload = {
        "head": sanitize_applied_rules(head_applied_rules),
        "tail": sanitize_applied_rules(tail_applied_rules),
    }
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(applied_payload, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
