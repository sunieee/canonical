#!/usr/bin/env python
# coding: utf-8
import multiprocessing as mp
import glob
import json
import os
from typing import Dict, List

import torch

import aggregation_single as agg

NUM_PROCESSES = 10

'''
单个进程 300%+，通常是因为：

- PyTorch 的 intra-op / inter-op 线程
- MKL / OpenBLAS / OpenMP 线程
- NumPy / SciPy 底层 BLAS 线程
- 所以“一个进程”仍可吃多个核。

因此多进程最大数量仅设置为 CPU 核数 / 2
'''

def _run_one_relation(relation: int):
    # Pool workers are daemonic; they cannot spawn children.
    # So DataLoader must run in-process in each worker.
    agg.args.max_worker_dataloader = 0
    agg.aggregate_single(relation)
    return int(relation)


def _get_all_relations() -> List[int]:
    return list(range(agg.dataset.num_relations()))


def _get_relation_test_counts() -> Dict[int, int]:
    relation_ids = agg.test_torch[:, 1].long().cpu()
    counts = torch.bincount(relation_ids, minlength=agg.dataset.num_relations())
    return {int(i): int(c) for i, c in enumerate(counts.tolist())}


def _merge_metric_files(metric_files: List[str], relation_test_counts: Dict[int, int]):
    rows = []
    for path in metric_files:
        with open(path, "r") as f:
            m = json.load(f)
        relation = int(m["relation"])
        test = m["test"]
        rows.append(
            {
                "relation": relation,
                "count": relation_test_counts.get(relation, 0),
                "mrr": float(test["mrr"]),
                "h1": float(test["h1"]),
                "h10": float(test["h10"]),
                "mrr_raw": float(test["mrr_raw"]),
                "h1_raw": float(test["h1_raw"]),
                "h10_raw": float(test["h10_raw"]),
            }
        )

    if not rows:
        return {
            "num_relations": 0,
            "macro": {},
            "weighted_by_test_triples": {},
        }

    keys = ["mrr", "h1", "h10", "mrr_raw", "h1_raw", "h10_raw"]
    weighted_rows = [r for r in rows if r["count"] > 0]
    total_weight = sum(r["count"] for r in weighted_rows)
    if total_weight > 0:
        weighted = {k: float(sum(r[k] * r["count"] for r in weighted_rows) / total_weight) for k in keys}
    else:
        weighted = {k: 0.0 for k in keys}

    return {
        "num_relations": len(rows),
        **weighted,
        "total_test_triples_used_for_weight": int(total_weight),
    }


def aggregate_multiple():
    # 在多进程 worker 内强制 DataLoader 单进程加载（num_workers=0）
    agg.args.max_worker_dataloader = 0

    relations = _get_all_relations()
    relation_test_counts = _get_relation_test_counts()

    print(f"Start relation sweep, total relations: {len(relations)}, processes: {min(NUM_PROCESSES, len(relations))}")

    success_relations = []
    failed_relations = {}

    if relations:
        with mp.get_context("fork").Pool(processes=min(NUM_PROCESSES, len(relations))) as pool:
            results = [pool.apply_async(_run_one_relation, (relation,)) for relation in relations]
            for relation, result in zip(relations, results):
                try:
                    success_relations.append(result.get())
                except Exception as e:
                    failed_relations[int(relation)] = str(e)

    metric_files = sorted(glob.glob(os.path.join(agg.args.experiment, "metric-*.json")))
    merged = _merge_metric_files(metric_files, relation_test_counts)

    final_result = {
        "experiment": agg.args.experiment,
        "model": agg.args.model,
        "dataset": agg.args.dataset,
        "success_relations": success_relations,
        "failed_relations": failed_relations,
        "summary": merged,
    }

    out_path = os.path.join(agg.args.experiment, "metrics-final.json")
    with open(out_path, "w") as f:
        json.dump(final_result, f, indent=4)

    print(f"Finished relation sweep. success={len(success_relations)}, failed={len(failed_relations)}")
    print(f"Final summary saved to {out_path}")
    return final_result


if __name__ == "__main__":
    result = aggregate_multiple()
    print(json.dumps(result["summary"], indent=2))
