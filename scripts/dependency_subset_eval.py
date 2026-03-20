#!/usr/bin/env python3
import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import pandas as pd


def load_relation_labels(dataset_dir):
    labels = {}
    with open(os.path.join(dataset_dir, "relation_ids.del"), "r", encoding="utf-8") as f:
        for line in f:
            idx, name = line.rstrip().split("\t")
            labels[int(idx)] = name
    return labels


def compute_relation_shape_stats(dataset_dir):
    sp = defaultdict(set)
    po = defaultdict(set)
    triples_by_rel = Counter()
    subj_by_rel = defaultdict(set)
    obj_by_rel = defaultdict(set)

    for split in ["train.del", "valid.del", "test.del"]:
        with open(os.path.join(dataset_dir, split), "r", encoding="utf-8") as f:
            for line in f:
                s, p, o = map(int, line.strip().split("\t"))
                sp[(s, p)].add(o)
                po[(p, o)].add(s)
                triples_by_rel[p] += 1
                subj_by_rel[p].add(s)
                obj_by_rel[p].add(o)

    rows = []
    for rel in sorted(triples_by_rel):
        sp_counts = [len(v) for (s, p), v in sp.items() if p == rel]
        po_counts = [len(v) for (p, o), v in po.items() if p == rel]
        rows.append(
            {
                "relation": rel,
                "num_triples": triples_by_rel[rel],
                "num_subjects": len(subj_by_rel[rel]),
                "num_objects": len(obj_by_rel[rel]),
                "avg_tails_per_sp": sum(sp_counts) / len(sp_counts),
                "avg_heads_per_po": sum(po_counts) / len(po_counts),
                "max_tails_per_sp": max(sp_counts),
                "max_heads_per_po": max(po_counts),
            }
        )
    return pd.DataFrame(rows)


def load_experiment_metrics(experiment_dir, relation_labels):
    rows = []
    for path in glob.glob(os.path.join(experiment_dir, "metric-*.json")):
        with open(path, "r") as f:
            m = json.load(f)
        rel = int(m["relation"])
        trial_path = os.path.join(experiment_dir, f"dependency-trial-{rel}.csv")
        trial_rows = (sum(1 for _ in open(trial_path, "r", encoding="utf-8")) - 1) if os.path.exists(trial_path) else 0
        rows.append(
            {
                "relation": rel,
                "rel_name": relation_labels.get(rel, str(rel)),
                "num_test": int(m["num_test_samples"]),
                "selected_stage": m["model_selection"]["selected_stage"],
                "dep_attempted": bool(m["model_selection"]["dependency_stage_attempted"]),
                "dep_accepted": m["model_selection"]["dependency_stage_accepted"],
                "rule_best_valid_raw": float(m["model_selection"]["rule_best_valid_combined_raw"]),
                "dep_best_valid_raw": (
                    None
                    if m["model_selection"]["dependency_best_valid_combined_raw"] is None
                    else float(m["model_selection"]["dependency_best_valid_combined_raw"])
                ),
                "rule_stage_test_raw": float(m["test_after_stage1"]["mrr_raw"]),
                "final_test_raw": float(m["test"]["mrr_raw"]),
                "num_rules": int(m["num_relation_rules"]),
                "num_deps": int(trial_rows),
            }
        )
    df = pd.DataFrame(rows).sort_values("relation")
    df["valid_gain"] = df["dep_best_valid_raw"].fillna(df["rule_best_valid_raw"]) - df["rule_best_valid_raw"]
    df["test_gain_vs_stage1"] = df["final_test_raw"] - df["rule_stage_test_raw"]
    df["dep_per_rule"] = df["num_deps"] / df["num_rules"].replace(0, 1)
    return df


def summarize_subset(df, name):
    if df.empty:
        return None
    w = df["num_test"]
    rule_raw = float((df["rule_stage_test_raw"] * w).sum() / w.sum())
    final_raw = float((df["final_test_raw"] * w).sum() / w.sum())
    return {
        "subset": name,
        "num_relations": int(len(df)),
        "num_test": int(w.sum()),
        "rule_stage_raw": rule_raw,
        "final_raw": final_raw,
        "gain_vs_stage1": final_raw - rule_raw,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute dependency-oriented subset evaluation and relation structure stats.")
    parser.add_argument("--dataset", default="codex-m")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--valid-gain-threshold", type=float, default=0.002)
    args = parser.parse_args()

    dataset_dir = os.path.join(args.data_root, args.dataset)
    relation_labels = load_relation_labels(dataset_dir)
    shape_df = compute_relation_shape_stats(dataset_dir)
    metrics_df = load_experiment_metrics(args.experiment_dir, relation_labels)
    merged = metrics_df.merge(shape_df, on="relation", how="left")

    subsets = {
        "all_relations": merged,
        "dep_accepted": merged[merged["dep_accepted"] == True],
        "dep_accepted_gain_gt_thresh": merged[
            (merged["dep_accepted"] == True) & (merged["valid_gain"] > float(args.valid_gain_threshold))
        ],
    }

    print("## Subset Evaluation")
    for name, df in subsets.items():
        summary = summarize_subset(df, name)
        if summary is None:
            continue
        print(json.dumps(summary, ensure_ascii=False))
        print("relations:", sorted(df["relation"].tolist()))

    print("\n## Relation Feature Medians")
    for name, df in subsets.items():
        if df.empty:
            continue
        med = df[
            [
                "num_triples",
                "num_subjects",
                "num_objects",
                "avg_tails_per_sp",
                "avg_heads_per_po",
                "num_rules",
                "num_deps",
                "dep_per_rule",
                "rule_best_valid_raw",
            ]
        ].median()
        print(f"[{name}]")
        print(med.to_string())

    print("\n## Accepted Relations")
    accepted = merged[merged["dep_accepted"] == True].copy()
    if not accepted.empty:
        print(
            accepted[
                [
                    "relation",
                    "rel_name",
                    "valid_gain",
                    "test_gain_vs_stage1",
                    "rule_best_valid_raw",
                    "final_test_raw",
                    "num_triples",
                    "avg_tails_per_sp",
                    "avg_heads_per_po",
                    "num_rules",
                    "num_deps",
                    "dep_per_rule",
                ]
            ]
            .sort_values("valid_gain", ascending=False)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
