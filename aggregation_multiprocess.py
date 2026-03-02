#!/usr/bin/env python
# coding: utf-8
import argparse
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from kge.kge import Config, Dataset


def get_parser():
    parser = argparse.ArgumentParser(description="Multiprocess launcher for single-relation aggregation")
    parser.add_argument("-d", "--dataset", default="codex-m", help="kge dataset name")
    parser.add_argument("-dev", "--device", default="cuda", help="cpu/cuda")
    parser.add_argument("--model", default="LinearAggregator", choices=["LinearAggregator", "NoisyOrAggregator"])
    parser.add_argument("--lr", type=float, required=True, help="Single learning rate")
    parser.add_argument("--max_epoch", type=int, required=True, help="Single epoch count")
    parser.add_argument("--pos", type=float, required=True, help="Single positive class weight")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--shuffle_train", action="store_true")
    parser.add_argument("--sign_constraint", action="store_true")
    parser.add_argument("--noisy_or_reg", action="store_true", default=False)
    parser.add_argument("--num_unseen", type=int, default=0)
    parser.add_argument("--max_worker_dataloader", type=int, default=max(os.cpu_count() - 1, 0))
    parser.add_argument("--max_processes", type=int, default=os.cpu_count(), help="Max concurrent relation processes")
    parser.add_argument("--relations", nargs="+", type=int, default=None, help="Optional subset of relation ids")
    parser.add_argument("--experiment_root", default=None, help="Output root directory")
    return parser


def build_rel_command(script_path, args, relation, output_json):
    cmd = [
        "python",
        script_path,
        "--dataset",
        args.dataset,
        "--device",
        args.device,
        "--relation",
        str(relation),
        "--model",
        args.model,
        "--lr",
        str(args.lr),
        "--max_epoch",
        str(args.max_epoch),
        "--pos",
        str(args.pos),
        "--batch_size",
        str(args.batch_size),
        "--max_worker_dataloader",
        str(args.max_worker_dataloader),
        "--num_unseen",
        str(args.num_unseen),
        "--experiment_root",
        args.experiment_root,
        "--output_json",
        output_json,
    ]
    if args.shuffle_train:
        cmd.append("--shuffle_train")
    if args.sign_constraint:
        cmd.append("--sign_constraint")
    if args.noisy_or_reg:
        cmd.append("--noisy_or_reg")
    return cmd


def run_one(command):
    proc = subprocess.run(command, capture_output=True, text=True)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "command": command,
    }


def weighted_mean(entries, key, count_key):
    num = 0.0
    den = 0
    for e in entries:
        c = int(e.get("counts", {}).get(count_key, 0))
        v = float(e.get(key, 0.0))
        num += v * c
        den += c
    return (num / den) if den > 0 else 0.0, den


def aggregate_results(results):
    valid_entries = [r for r in results if (not r.get("skipped", False)) and r.get("result") is not None]
    if len(valid_entries) == 0:
        return {
            "valid": {"mrr": 0.0, "h1": 0.0, "h10": 0.0, "mrr_raw": 0.0, "h1_raw": 0.0, "h10_raw": 0.0},
            "test": {"mrr": 0.0, "h1": 0.0, "h10": 0.0, "mrr_raw": 0.0, "h1_raw": 0.0, "h10_raw": 0.0},
            "n_relations": 0,
        }

    def collect(split):
        # split: valid/test
        head_mrr, n_h = weighted_mean([r["result"][split]["head"] for r in valid_entries], "mrr", "test_head")
        tail_mrr, n_t = weighted_mean([r["result"][split]["tail"] for r in valid_entries], "mrr", "test_tail")
        head_h1, _ = weighted_mean([r["result"][split]["head"] for r in valid_entries], "h1", "test_head")
        tail_h1, _ = weighted_mean([r["result"][split]["tail"] for r in valid_entries], "h1", "test_tail")
        head_h10, _ = weighted_mean([r["result"][split]["head"] for r in valid_entries], "h10", "test_head")
        tail_h10, _ = weighted_mean([r["result"][split]["tail"] for r in valid_entries], "h10", "test_tail")

        head_mrr_raw, _ = weighted_mean([r["result"][split]["head_raw"] for r in valid_entries], "mrr", "test_head")
        tail_mrr_raw, _ = weighted_mean([r["result"][split]["tail_raw"] for r in valid_entries], "mrr", "test_tail")
        head_h1_raw, _ = weighted_mean([r["result"][split]["head_raw"] for r in valid_entries], "h1", "test_head")
        tail_h1_raw, _ = weighted_mean([r["result"][split]["tail_raw"] for r in valid_entries], "h1", "test_tail")
        head_h10_raw, _ = weighted_mean([r["result"][split]["head_raw"] for r in valid_entries], "h10", "test_head")
        tail_h10_raw, _ = weighted_mean([r["result"][split]["tail_raw"] for r in valid_entries], "h10", "test_tail")

        return {
            "mrr": (head_mrr + tail_mrr) / 2,
            "h1": (head_h1 + tail_h1) / 2,
            "h10": (head_h10 + tail_h10) / 2,
            "mrr_raw": (head_mrr_raw + tail_mrr_raw) / 2,
            "h1_raw": (head_h1_raw + tail_h1_raw) / 2,
            "h10_raw": (head_h10_raw + tail_h10_raw) / 2,
            "head_count": n_h,
            "tail_count": n_t,
        }

    return {
        "valid": collect("valid"),
        "test": collect("test"),
        "n_relations": len(valid_entries),
    }


def main():
    args = get_parser().parse_args()

    run_time = datetime.now().strftime("%m%d-%H%M%S")
    args.experiment_root = args.experiment_root or f"./{args.dataset}/mp-run-{run_time}"
    os.makedirs(args.experiment_root, exist_ok=True)
    rel_result_dir = os.path.join(args.experiment_root, "relations")
    os.makedirs(rel_result_dir, exist_ok=True)

    cfg_path = os.path.join(args.experiment_root, "launcher_config.json")
    with open(cfg_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    c = kge.Config()
    c.set("dataset.name", args.dataset)
    dataset = kge.Dataset.create(c)

    relations = args.relations if args.relations is not None else list(range(dataset.num_relations()))

    script_path = os.path.join(os.path.dirname(__file__), "aggregation.py")
    futures = {}
    process_outputs = []

    with ProcessPoolExecutor(max_workers=args.max_processes) as executor:
        for rel in relations:
            out_json = os.path.join(rel_result_dir, f"relation_{rel}.json")
            cmd = build_rel_command(script_path, args, rel, out_json)
            futures[executor.submit(run_one, cmd)] = (rel, out_json)

        for fut in as_completed(futures):
            rel, out_json = futures[fut]
            run_info = fut.result()
            run_info["relation"] = rel
            run_info["output_json"] = out_json
            process_outputs.append(run_info)

    process_log = os.path.join(args.experiment_root, "process_outputs.json")
    with open(process_log, "w") as f:
        json.dump(process_outputs, f, indent=2)

    relation_results = []
    for rel in relations:
        p = os.path.join(rel_result_dir, f"relation_{rel}.json")
        if os.path.exists(p):
            with open(p) as f:
                relation_results.append(json.load(f))
        else:
            relation_results.append({"relation": rel, "skipped": True, "reason": "output_missing"})

    summary = {
        "config": vars(args),
        "relations": relation_results,
        "aggregate": aggregate_results(relation_results),
    }

    summary_path = os.path.join(args.experiment_root, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Done. Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
