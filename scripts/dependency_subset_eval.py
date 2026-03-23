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


def compute_relation_shape_stats(dataset_dir, splits=None):
    if splits is None:
        splits = ["train.del"]
    sp = defaultdict(set)
    po = defaultdict(set)
    triples_by_rel = Counter()
    subj_by_rel = defaultdict(set)
    obj_by_rel = defaultdict(set)

    for split in splits:
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
        "relative_gain_vs_stage1": (final_raw - rule_raw) / rule_raw if rule_raw else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute fair dependency-oriented subset evaluation from static relation features.")
    parser.add_argument("-d", "--dataset", default="codex-m")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--shape-splits", nargs="+", default=["train.del"], help="Splits used to compute relation shape features.")
    parser.add_argument("--subset-min-train", type=int, default=0)
    parser.add_argument("--subset-min-objects", type=int, default=100)
    parser.add_argument("--subset-min-avg-tails", type=float, default=1.1)
    parser.add_argument("--subset-max-avg-tails", type=float, default=1.8)
    parser.add_argument("--emit-strict-subset", action="store_true")
    parser.add_argument("--strict-min-train", type=int, default=2000)
    args = parser.parse_args()

    dataset_dir = os.path.join(args.data_root, args.dataset)
    relation_labels = load_relation_labels(dataset_dir)
    shape_df = compute_relation_shape_stats(dataset_dir, splits=args.shape_splits)
    metrics_df = load_experiment_metrics(args.experiment_dir, relation_labels)
    merged = metrics_df.merge(shape_df, on="relation", how="left")

    train_only_subset = merged[
        (merged["num_triples"] >= int(args.subset_min_train))
        & (merged["num_objects"] >= int(args.subset_min_objects))
        & (merged["avg_tails_per_sp"] >= float(args.subset_min_avg_tails))
        & (merged["avg_tails_per_sp"] <= float(args.subset_max_avg_tails))
    ]

    subsets = {
        "all_relations": merged,
        "dependency_friendly_train_only": train_only_subset,
    }
    if args.emit_strict_subset:
        subsets["dependency_friendly_train_only_strict"] = train_only_subset[
            train_only_subset["num_triples"] >= int(args.strict_min_train)
        ]

    print("## Subset Evaluation")
    print(
        json.dumps(
            {
                "shape_splits": args.shape_splits,
                "dependency_friendly_rule": {
                    "min_train": int(args.subset_min_train),
                    "min_objects": int(args.subset_min_objects),
                    "min_avg_tails_per_sp": float(args.subset_min_avg_tails),
                    "max_avg_tails_per_sp": float(args.subset_max_avg_tails),
                },
                "strict_min_train": int(args.strict_min_train) if args.emit_strict_subset else None,
            },
            ensure_ascii=False,
        )
    )
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

    print("\n## Dependency-Friendly Relations")
    dep_subset = subsets["dependency_friendly_train_only"].copy()
    if not dep_subset.empty:
        print(
            dep_subset[
                [
                    "relation",
                    "rel_name",
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
            .sort_values("test_gain_vs_stage1", ascending=False)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
