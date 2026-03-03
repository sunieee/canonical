#!/usr/bin/env python
# coding: utf-8
import argparse
import json
import multiprocessing as mp
import os
from datetime import datetime

import kge

from aggregation_single_relation import get_parser as get_single_parser
from aggregation_single_relation import run_single_relation_experiment


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config-base.json")
    parser.add_argument("-d", "--dataset", default="codex-m")
    parser.add_argument("-dev", "--device", default="cuda")
    parser.add_argument("--max_processes", type=int, default=os.cpu_count())
    parser.add_argument("--only_first_relation", action="store_true", default=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--model", default="LinearAggregator", choices=["LinearAggregator", "NoisyOrAggregator"])
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_epoch", type=int, default=20)
    parser.add_argument("--pos", type=float, default=15)
    parser.add_argument("--sign_constraint", action="store_true")
    parser.add_argument("--shuffle_train", action="store_true")
    parser.add_argument("--max_worker_dataloader", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    return parser


def build_relation_list(dataset_name, only_first_relation=True):
    c = kge.Config()
    c.set("dataset.name", dataset_name)
    ds = kge.Dataset.create(c)
    relations = list(range(ds.num_relations()))
    if only_first_relation:
        return relations[:1]
    return relations


def to_single_relation_namespace(global_args, relation, output_dir):
    single_parser = get_single_parser()
    single_args = single_parser.parse_args([])

    single_args.config = global_args.config
    single_args.dataset = global_args.dataset
    single_args.device = global_args.device
    single_args.relation = relation
    single_args.max_worker_dataloader = global_args.max_worker_dataloader
    single_args.model = global_args.model
    single_args.shuffle_train = global_args.shuffle_train
    single_args.batch_size = global_args.batch_size
    single_args.lr = global_args.lr
    single_args.max_epoch = global_args.max_epoch
    single_args.pos = global_args.pos
    single_args.sign_constraint = global_args.sign_constraint
    single_args.output_dir = output_dir
    return single_args


def worker_run(single_args):
    return run_single_relation_experiment(single_args)


def aggregate_results(results):
    ok_results = [r for r in results if r and "metrics" in r and r.get("metrics") is not None]
    if len(ok_results) == 0:
        return {
            "num_relations": len(results),
            "num_valid_results": 0,
            "avg_test_combined_mrr": 0.0,
            "avg_test_combined_mrr_raw": 0.0,
        }

    avg_test_mrr = sum(r["metrics"]["test_combined_mrr"] for r in ok_results) / len(ok_results)
    avg_test_mrr_raw = sum(r["metrics"]["test_combined_mrr_raw"] for r in ok_results) / len(ok_results)
    return {
        "num_relations": len(results),
        "num_valid_results": len(ok_results),
        "avg_test_combined_mrr": avg_test_mrr,
        "avg_test_combined_mrr_raw": avg_test_mrr_raw,
    }


def run_multiprocess(args=None):
    if args is None:
        args = get_parser().parse_args()

    if args.output_dir is None:
        ts = datetime.now().strftime("%m%d-%H%M%S")
        args.output_dir = f"./{args.dataset}/exp-mp-{ts}"
    os.makedirs(args.output_dir, exist_ok=True)

    relations = build_relation_list(args.dataset, only_first_relation=args.only_first_relation)

    all_single_args = [to_single_relation_namespace(args, rel, args.output_dir) for rel in relations]

    max_processes = min(max(1, args.max_processes or os.cpu_count() or 1), os.cpu_count() or 1, len(all_single_args))

    with mp.Pool(processes=max_processes) as pool:
        results = pool.map(worker_run, all_single_args)

    summary = aggregate_results(results)
    summary["relations"] = relations
    summary["output_dir"] = args.output_dir

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


if __name__ == "__main__":
    run_multiprocess()
