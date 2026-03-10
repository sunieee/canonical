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


def main():
    parser = argparse.ArgumentParser(
        description="AnyBURL Apply-like output using PyClause ranking + rule features"
    )
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--rules", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-explanations", type=int, default=200)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--worker-threads", type=int, default=-1)
    parser.add_argument("--aggregation", default="maxplus")
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
    if args.worker_threads is not None and int(args.worker_threads) > 0:
        opts.set("loader.num_threads", int(args.worker_threads))

    # ranking configuration
    opts.set("ranking_handler.collect_rules", True)
    opts.set("ranking_handler.topk", args.topk)
    aggregation = args.aggregation
    if aggregation in ("maxplus-explanation", "maxplus-explanation-stdout"):
        aggregation = "maxplus"
    opts.set("ranking_handler.aggregation_function", aggregation)
    opts.set("ranking_handler.num_top_rules", args.max_explanations)
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

    # applied rules from PyClause (rule ids align with rules file line numbers, 1-based)
    head_applied_rules = ranker.get_applied_rules(direction="head")
    tail_applied_rules = ranker.get_applied_rules(direction="tail")

    applied_payload = {
        "head": sanitize_applied_rules(head_applied_rules),
        "tail": sanitize_applied_rules(tail_applied_rules),
    }
    os.makedirs(args.output.rsplit("/", 1)[0], exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(applied_payload, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
