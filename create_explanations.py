#!/usr/bin/env python
# coding: utf-8

import argparse
from collections import defaultdict
import copy
import json
import os
import pickle
from tqdm import tqdm
from typing import Dict, List, Tuple

from c_clause import Loader, RankingHandler
from clause import Options


def read_ids(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().splitlines()
    return [line.split("\t")[1] for line in raw]


def read_triples(path: str, ent_to_id: Dict[str, int], rel_to_id: Dict[str, int]) -> List[Tuple[int, int, int]]:
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s, p, o = line.split("\t")
            triples.append((ent_to_id[s], rel_to_id[p], ent_to_id[o]))
    return triples


def build_rule_features_from_rules_file(rules_path: str, allowed_rule_ids=None):
    """
    全局 rule id 采用 rules 文件行号（1-based）。
    与 PyClause 内部 rule id 保持一致。
    """
    rule_features = {}
    with open(rules_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if allowed_rule_ids is not None and line_no not in allowed_rule_ids:
                continue
            parts = line.rstrip("\n").split("\t", 3)
            if len(parts) < 4:
                continue
            global_id = line_no
            num_preds = parts[0]
            num_true = parts[1]
            rule_str = parts[3]
            rule_features[global_id] = [num_preds, num_true, rule_str]
    return rule_features


def build_rule_map_from_rule_features(rule_features, relation_to_id):
    rule_map = defaultdict(list)
    for rule_id in sorted(rule_features.keys()):
        rule = rule_features[rule_id]
        if len(rule) < 3:
            continue
        rule_str = str(rule[2])
        head = rule_str.split(" <= ", 1)[0].strip()
        if "(" not in head:
            continue
        rel_token = head.split("(", 1)[0].strip()
        rel_id = relation_to_id.get(rel_token)
        if rel_id is not None:
            rule_map[rel_id].append(int(rule_id))
    return dict(rule_map)


def pyclause_apply(
    data_path: str,
    filter_path: str,
    target_path: str,
    rules_path: str,
    topk: int,
    loader_threads: int,
    ranking_threads: int,
    aggregation_function: str,
    disable_b: bool,
    disable_u_d: bool,
    disable_u_c: bool,
    disable_zero: bool,
    disable_u_xxc: bool,
    disable_u_xxd: bool,
    b_max_length: int,
    num_unseen: int,
    d_weight: float,
    min_correct_predictions: int,
    max_explanations: int,
):
    opts = Options()
    opts.set("ranking_handler.aggregation_function", aggregation_function)
    opts.set("ranking_handler.collect_rules", True)
    opts.set("ranking_handler.topk", int(topk))
    opts.set("ranking_handler.num_threads", int(ranking_threads))
    opts.set("ranking_handler.num_top_rules", int(max_explanations))

    opts.set("loader.load_b_rules", not disable_b)
    opts.set("loader.load_zero_rules", not disable_zero)
    opts.set("loader.load_u_d_rules", not disable_u_d)
    opts.set("loader.load_u_c_rules", not disable_u_c)
    opts.set("loader.load_u_xxc_rules", not disable_u_xxc)
    opts.set("loader.load_u_xxd_rules", not disable_u_xxd)
    opts.set("loader.b_max_length", int(b_max_length))
    opts.set("loader.b_min_support", int(min_correct_predictions))
    opts.set("loader.c_min_support", int(min_correct_predictions))
    opts.set("loader.num_unseen", int(num_unseen))
    opts.set("loader.d_weight", float(d_weight))
    opts.set("loader.num_threads", int(loader_threads))

    loader = Loader(options=opts.get("loader"))
    loader.load_data(data=data_path, filter=filter_path, target=target_path)
    loader.load_rules(rules=rules_path)

    ranker = RankingHandler(options=opts.get("ranking_handler"))
    ranker.calculate_ranking(loader=loader)

    # int 索引版本，便于后续直接写入 processed_*.pkl
    head_ranking = ranker.get_ranking(direction="head", as_string=False)
    tail_ranking = ranker.get_ranking(direction="tail", as_string=False)
    head_rules = ranker.get_rules(direction="head", as_string=False)
    tail_rules = ranker.get_rules(direction="tail", as_string=False)
    return head_ranking, tail_ranking, head_rules, tail_rules


def get_active_global_rule_ids(
    data_path: str,
    filter_path: str,
    target_path: str,
    rules_path: str,
    loader_threads: int,
    disable_b: bool,
    disable_u_d: bool,
    disable_u_c: bool,
    disable_zero: bool,
    disable_u_xxc: bool,
    disable_u_xxd: bool,
    b_max_length: int,
    num_unseen: int,
    d_weight: float,
    min_correct_predictions: int,
):
    opts = Options()
    opts.set("loader.load_b_rules", not disable_b)
    opts.set("loader.load_zero_rules", not disable_zero)
    opts.set("loader.load_u_d_rules", not disable_u_d)
    opts.set("loader.load_u_c_rules", not disable_u_c)
    opts.set("loader.load_u_xxc_rules", not disable_u_xxc)
    opts.set("loader.load_u_xxd_rules", not disable_u_xxd)
    opts.set("loader.b_max_length", int(b_max_length))
    opts.set("loader.b_min_support", int(min_correct_predictions))
    opts.set("loader.c_min_support", int(min_correct_predictions))
    opts.set("loader.num_unseen", int(num_unseen))
    opts.set("loader.d_weight", float(d_weight))
    opts.set("loader.num_threads", int(loader_threads))

    loader = Loader(options=opts.get("loader"))
    loader.load_data(data=data_path, filter=filter_path, target=target_path)
    loader.load_rules(rules=rules_path)

    # PyClause rule_index() 是 1-based（0 号位为空占位）。
    # 对齐 process.sh 行为：只保留当前配置下真正启用并被读取的规则。
    rule_index = loader.rule_index()
    active_rule_ids = set()
    for rid in range(1, len(rule_index)):
        if rule_index[rid] != "":
            active_rule_ids.add(rid)
    return active_rule_ids


def _convert_rule_ids_to_global(rule_ids: List[int]) -> List[int]:
    # 全局 rule id 与 PyClause 内部 id 一致（均为 1-based 行号）
    converted = []
    for rid in rule_ids:
        rid = int(rid)
        if rid > 0:
            converted.append(rid)
    return converted


def build_processed_from_apply(
    target_triples: List[Tuple[int, int, int]],
    head_ranking,
    tail_ranking,
    head_rules,
    tail_rules,
    max_explanations: int,
):
    processed = {}
    processed_sp = {}
    processed_po = {}
    longest = 0

    # 缓存：同一个 (p,s) / (p,o) 在 target 中会重复出现，避免重复构造 candidates/rules
    tails_cache = {}
    heads_cache = {}

    def get_tails_meta(p: int, s: int):
        nonlocal longest
        key = (p, s)
        if key in tails_cache:
            return tails_cache[key]

        tails_scored = tail_ranking.get(p, {}).get(s, [])
        tails_candidates = [int(x[0]) for x in tails_scored]
        tails_rules_map = tail_rules.get(p, {}).get(s, {})
        tails_rules = []
        tails_pos = {}
        for idx, cand in enumerate(tails_candidates):
            tails_pos[cand] = idx
            rids = _convert_rule_ids_to_global(tails_rules_map.get(cand, []))
            if max_explanations > 0:
                rids = rids[:max_explanations]
            if len(rids) > longest:
                longest = len(rids)
            tails_rules.append(rids)

        meta = (tails_candidates, tails_rules, tails_pos)
        tails_cache[key] = meta
        return meta

    def get_heads_meta(p: int, o: int):
        nonlocal longest
        key = (p, o)
        if key in heads_cache:
            return heads_cache[key]

        heads_scored = head_ranking.get(p, {}).get(o, [])
        heads_candidates = [int(x[0]) for x in heads_scored]
        heads_rules_map = head_rules.get(p, {}).get(o, {})
        heads_rules = []
        heads_pos = {}
        for idx, cand in enumerate(heads_candidates):
            heads_pos[cand] = idx
            rids = _convert_rule_ids_to_global(heads_rules_map.get(cand, []))
            if max_explanations > 0:
                rids = rids[:max_explanations]
            if len(rids) > longest:
                longest = len(rids)
            heads_rules.append(rids)

        meta = (heads_candidates, heads_rules, heads_pos)
        heads_cache[key] = meta
        return meta

    for s, p, o in tqdm(target_triples):
        tails_candidates, tails_rules, tails_pos = get_tails_meta(p, s)
        heads_candidates, heads_rules, heads_pos = get_heads_meta(p, o)

        raw_meta = {
            "heads": {"candidates": heads_candidates, "rules": heads_rules},
            "tails": {"candidates": tails_candidates, "rules": tails_rules},
        }

        processed[(s, p, o)] = raw_meta

        # 与 preprocess_explanations.py 保持一致
        if (s, p) in processed_sp:
            if o in tails_pos:
                processed_sp[(s, p)]["candidates"].append(o)
                idx_o = tails_pos[o]
                processed_sp[(s, p)]["rules"].append(copy.copy(raw_meta["tails"]["rules"][idx_o]))
        else:
            processed_sp[(s, p)] = {
                "candidates": copy.deepcopy(raw_meta["tails"]["candidates"]),
                "rules": copy.deepcopy(raw_meta["tails"]["rules"]),
            }

        if (p, o) in processed_po:
            if s in heads_pos:
                processed_po[(p, o)]["candidates"].append(s)
                idx_s = heads_pos[s]
                processed_po[(p, o)]["rules"].append(copy.copy(raw_meta["heads"]["rules"][idx_s]))
        else:
            processed_po[(p, o)] = {
                "candidates": copy.deepcopy(raw_meta["heads"]["candidates"]),
                "rules": copy.deepcopy(raw_meta["heads"]["rules"]),
            }

    print(f"Longest rule set for a candidate: {longest}")
    return processed, processed_sp, processed_po


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def dump_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use PyClause to apply rules for train/valid/test and produce canonical processed explanations."
    )
    parser.add_argument("--dataset", "-d", default="codex-m", help="Dataset folder under canonical/, e.g. codex-m")
    parser.add_argument("--data_dir", default="", help="Absolute or relative dataset dir. Overrides --dataset if set.")
    parser.add_argument("--rules_file", default="", help="Path to rules file. Default: <data_dir>/rules/rules-1000")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--aggregation_function", default="maxplus")
    parser.add_argument("--loader_threads", type=int, default=os.cpu_count())
    parser.add_argument("--ranking_threads", type=int, default=-1)

    # 对齐 explanations/process.sh：默认启用 C/B，关闭 Acyclic2/Zero。
    parser.add_argument("--disable_b", action="store_true", default=False)
    parser.add_argument("--disable_u_d", action="store_true", default=True)
    parser.add_argument("--enable_u_d", action="store_false", dest="disable_u_d")
    parser.add_argument("--disable_u_c", action="store_true", default=False)
    parser.add_argument("--disable_zero", action="store_true", default=True)
    parser.add_argument("--enable_zero", action="store_false", dest="disable_zero")
    parser.add_argument("--disable_u_xxc", action="store_true", default=True)
    parser.add_argument("--enable_u_xxc", action="store_false", dest="disable_u_xxc")
    parser.add_argument("--disable_u_xxd", action="store_true", default=True)
    parser.add_argument("--enable_u_xxd", action="store_false", dest="disable_u_xxd")
    parser.add_argument("--b_max_length", type=int, default=-1)
    parser.add_argument("--num_unseen", type=int, default=5)
    parser.add_argument("--d_weight", type=float, default=0.1)
    parser.add_argument("--min_correct_predictions", type=int, default=5)
    parser.add_argument("--max_explanations", type=int, default=200)

    parser.add_argument(
        "--dump_raw_apply_json",
        action="store_true",
        help="Also dump raw ranking/rules json into expl/explanations for debugging.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir if args.data_dir else os.path.join(script_dir, args.dataset)
    data_dir = os.path.abspath(data_dir)

    rules_file = args.rules_file if args.rules_file else os.path.join(data_dir, "rules", "rules-1000")
    rules_file = os.path.abspath(rules_file)

    expl_dir = os.path.join(data_dir, "expl")
    explanations_dir = os.path.join(expl_dir, "explanations")
    processed_dir = os.path.join(expl_dir, "explanations-processed")
    ensure_dir(explanations_dir)
    ensure_dir(processed_dir)

    ent_ids = read_ids(os.path.join(data_dir, "entity_ids.del"))
    rel_ids = read_ids(os.path.join(data_dir, "relation_ids.del"))
    ent_to_id = {v: i for i, v in enumerate(ent_ids)}
    rel_to_id = {v: i for i, v in enumerate(rel_ids)}

    empty_filter = os.path.join(data_dir, "empty.txt")

    active_rule_ids = get_active_global_rule_ids(
        data_path=os.path.join(data_dir, "train.txt"),
        filter_path=empty_filter,
        target_path=os.path.join(data_dir, "train.txt"),
        rules_path=rules_file,
        loader_threads=args.loader_threads,
        disable_b=args.disable_b,
        disable_u_d=args.disable_u_d,
        disable_u_c=args.disable_u_c,
        disable_zero=args.disable_zero,
        disable_u_xxc=args.disable_u_xxc,
        disable_u_xxd=args.disable_u_xxd,
        b_max_length=args.b_max_length,
        num_unseen=args.num_unseen,
        d_weight=args.d_weight,
        min_correct_predictions=args.min_correct_predictions,
    )

    rule_features = build_rule_features_from_rules_file(rules_file, allowed_rule_ids=active_rule_ids)
    rule_map = build_rule_map_from_rule_features(rule_features, rel_to_id)

    dump_pickle(rule_features, os.path.join(processed_dir, "rule_features.pkl"))
    dump_pickle(rule_map, os.path.join(processed_dir, "rule_map.pkl"))
    print(f"Saved rule_features.pkl with {len(rule_features)} global rules")
    print(f"Saved rule_map.pkl with {len(rule_map)} relations")

    split_plan = [
        {
            "name": "train",
            "data": os.path.join(data_dir, "train.txt"),
            "filter": empty_filter,
            "target": os.path.join(data_dir, "train.txt"),
        },
        {
            "name": "valid",
            "data": os.path.join(data_dir, "train.txt"),
            "filter": empty_filter,
            "target": os.path.join(data_dir, "valid.txt"),
        },
        {
            "name": "test",
            "data": os.path.join(data_dir, "train.txt"),
            "filter": os.path.join(data_dir, "valid.txt"),
            "target": os.path.join(data_dir, "test.txt"),
        },
    ]

    for cfg in split_plan:
        split = cfg["name"]
        print(f"\n===== Processing {split} =====")
        head_ranking, tail_ranking, head_rules, tail_rules = pyclause_apply(
            data_path=cfg["data"],
            filter_path=cfg["filter"],
            target_path=cfg["target"],
            rules_path=rules_file,
            topk=args.topk,
            loader_threads=args.loader_threads,
            ranking_threads=args.ranking_threads,
            aggregation_function=args.aggregation_function,
            disable_b=args.disable_b,
            disable_u_d=args.disable_u_d,
            disable_u_c=args.disable_u_c,
            disable_zero=args.disable_zero,
            disable_u_xxc=args.disable_u_xxc,
            disable_u_xxd=args.disable_u_xxd,
            b_max_length=args.b_max_length,
            num_unseen=args.num_unseen,
            d_weight=args.d_weight,
            min_correct_predictions=args.min_correct_predictions,
            max_explanations=args.max_explanations,
        )

        if args.dump_raw_apply_json:
            raw_path = os.path.join(explanations_dir, f"{split}-pyclause-apply.json")
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "head_ranking": head_ranking,
                        "tail_ranking": tail_ranking,
                        "head_rules": head_rules,
                        "tail_rules": tail_rules,
                    },
                    f,
                    ensure_ascii=False,
                )
            print(f"Dumped raw apply json: {raw_path}")

        print(f"Building processed explanations for {split}...")
        target_triples = read_triples(cfg["target"], ent_to_id=ent_to_id, rel_to_id=rel_to_id)
        processed, processed_sp, processed_po = build_processed_from_apply(
            target_triples=target_triples,
            head_ranking=head_ranking,
            tail_ranking=tail_ranking,
            head_rules=head_rules,
            tail_rules=tail_rules,
            max_explanations=args.max_explanations,
        )

        dump_pickle(processed, os.path.join(processed_dir, f"processed_explanations_{split}.pkl"))
        dump_pickle(processed_sp, os.path.join(processed_dir, f"processed_sp_{split}.pkl"))
        dump_pickle(processed_po, os.path.join(processed_dir, f"processed_po_{split}.pkl"))
        print(f"Saved processed files for {split}")

    print("\nAll done.")
    print(f"Processed explanations dir: {processed_dir}")


if __name__ == "__main__":
    main()
