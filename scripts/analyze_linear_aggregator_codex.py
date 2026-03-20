#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NEAR_ZERO_THRESHOLDS = [1e-6, 1e-4, 1e-3, 1e-2]


@dataclass
class ExperimentData:
    base_dir: str
    metrics: pd.DataFrame
    rules_final: pd.DataFrame
    rules_dependency_stage: pd.DataFrame
    dependencies: pd.DataFrame


def load_metrics(base_dir: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(base_dir, "metric-*.json"))):
        with open(path) as f:
            m = json.load(f)
        rows.append(
            {
                "relation": int(m["relation"]),
                "selected_stage": m["model_selection"]["selected_stage"],
                "dependency_stage_attempted": bool(m["model_selection"]["dependency_stage_attempted"]),
                "dependency_stage_accepted": (
                    None
                    if m["model_selection"]["dependency_stage_accepted"] is None
                    else bool(m["model_selection"]["dependency_stage_accepted"])
                ),
                "rule_best_valid_combined_raw": float(m["model_selection"]["rule_best_valid_combined_raw"]),
                "dependency_best_valid_combined_raw": (
                    np.nan
                    if m["model_selection"]["dependency_best_valid_combined_raw"] is None
                    else float(m["model_selection"]["dependency_best_valid_combined_raw"])
                ),
                "num_relation_rules": int(m["num_relation_rules"]),
                "num_relation_dependencies": int(m["num_relation_dependencies"]),
                "test_mrr": float(m["test"]["mrr"]),
                "test_mrr_raw": float(m["test"]["mrr_raw"]),
            }
        )
    return pd.DataFrame(rows).sort_values("relation").reset_index(drop=True)


def load_rule_weights(base_dir: str, metrics_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    accepted_relations = set(metrics_df.loc[metrics_df["dependency_stage_accepted"] == True, "relation"].tolist())

    for path in sorted(glob.glob(os.path.join(base_dir, "weight-*.csv"))):
        relation = int(os.path.basename(path).split("-")[1].split(".")[0])
        df = pd.read_csv(path)
        df["relation"] = relation
        df["abs_original"] = df["original"].abs()
        df["abs_trained"] = df["trained"].abs()
        df["delta"] = df["trained"] - df["original"]
        df["abs_delta"] = df["delta"].abs()
        df["final_stage"] = "dependency" if relation in accepted_relations else "rule_only"
        frames.append(df)

    all_rules = pd.concat(frames, ignore_index=True)
    return all_rules


def load_dependency_stage_rule_weights(base_dir: str) -> pd.DataFrame:
    frames = []
    for dep_path in sorted(glob.glob(os.path.join(base_dir, "dependency-*.csv"))):
        relation = int(os.path.basename(dep_path).split("-")[1].split(".")[0])
        weight_path = os.path.join(base_dir, f"weight-{relation}.csv")
        if not os.path.exists(weight_path):
            continue
        dep_df = pd.read_csv(dep_path)
        if dep_df.empty:
            continue
        rule_ids = sorted(set(dep_df["rule1ID"].tolist()) | set(dep_df["rule2ID"].tolist()))
        # This will be overwritten later for rejected relations because the saved weight-*.csv is final-stage.
        # We keep the table for accepted relations and for rule-level lookups when the stages coincide.
        weight_df = pd.read_csv(weight_path)
        weight_df = weight_df[weight_df["ruleID"].isin(rule_ids)].copy()
        weight_df["relation"] = relation
        frames.append(weight_df)
    if not frames:
        return pd.DataFrame(columns=["ruleID", "original", "trained", "train_hit_count", "relation"])
    df = pd.concat(frames, ignore_index=True)
    df["abs_trained"] = df["trained"].abs()
    return df


def load_dependencies(base_dir: str, metrics_df: pd.DataFrame, rules_final_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = metrics_df[
        ["relation", "selected_stage", "dependency_stage_attempted", "dependency_stage_accepted"]
    ].copy()
    rule_lookup = rules_final_df[["relation", "ruleID", "trained", "original"]].copy()
    rule1_lookup = rule_lookup.rename(
        columns={"ruleID": "rule1ID", "trained": "rule1_final_trained", "original": "rule1_final_original"}
    )
    rule2_lookup = rule_lookup.rename(
        columns={"ruleID": "rule2ID", "trained": "rule2_final_trained", "original": "rule2_final_original"}
    )

    frames = []
    for path in sorted(glob.glob(os.path.join(base_dir, "dependency-*.csv"))):
        relation = int(os.path.basename(path).split("-")[1].split(".")[0])
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["relation"] = relation
        df["abs_effective_trained"] = df["effective_trained"].abs()
        df["abs_raw_trained"] = df["raw_trained"].abs()
        frames.append(df)
    dep_df = pd.concat(frames, ignore_index=True)
    dep_df = dep_df.merge(metric_cols, on="relation", how="left")
    dep_df = dep_df.merge(rule1_lookup, on=["relation", "rule1ID"], how="left")
    dep_df = dep_df.merge(rule2_lookup, on=["relation", "rule2ID"], how="left")
    dep_df["rule_min_abs_final_trained"] = dep_df[["rule1_final_trained", "rule2_final_trained"]].abs().min(axis=1)
    dep_df["rule_max_abs_final_trained"] = dep_df[["rule1_final_trained", "rule2_final_trained"]].abs().max(axis=1)
    dep_df["rule_mean_abs_final_trained"] = (
        dep_df["rule1_final_trained"].abs() + dep_df["rule2_final_trained"].abs()
    ) / 2.0
    dep_df["semantic_match"] = np.where(
        dep_df["type"] == "synergy",
        dep_df["effective_trained"] >= 0,
        dep_df["effective_trained"] <= 0,
    )
    dep_df["semantic_violation"] = ~dep_df["semantic_match"]
    dep_df["both_rules_near_zero_1e3"] = (
        dep_df["rule1_final_trained"].abs().fillna(np.inf) <= 1e-3
    ) & (dep_df["rule2_final_trained"].abs().fillna(np.inf) <= 1e-3)
    dep_df["any_rule_near_zero_1e3"] = (
        dep_df["rule1_final_trained"].abs().fillna(np.inf) <= 1e-3
    ) | (dep_df["rule2_final_trained"].abs().fillna(np.inf) <= 1e-3)
    return dep_df


def summarize_rules(rules_df: pd.DataFrame) -> dict:
    summary = {
        "num_rules": int(len(rules_df)),
        "original_mean": float(rules_df["original"].mean()),
        "original_median": float(rules_df["original"].median()),
        "trained_mean": float(rules_df["trained"].mean()),
        "trained_median": float(rules_df["trained"].median()),
        "corr_original_trained": float(rules_df["original"].corr(rules_df["trained"], method="spearman")),
        "corr_abs_original_abs_trained": float(rules_df["abs_original"].corr(rules_df["abs_trained"], method="spearman")),
    }
    for th in NEAR_ZERO_THRESHOLDS:
        key = f"trained_abs_le_{th:g}"
        summary[key] = float((rules_df["abs_trained"] <= th).mean())
    # "Originally non-trivial but later near zero"
    for cutoff in [0.01, 0.05, 0.1]:
        mask = rules_df["original"] >= cutoff
        if int(mask.sum()) == 0:
            continue
        summary[f"original_ge_{cutoff:g}_and_trained_abs_le_1e-3"] = float(
            ((rules_df["abs_trained"] <= 1e-3) & mask).sum() / mask.sum()
        )
    return summary


def summarize_dependencies(dep_df: pd.DataFrame) -> dict:
    summary = {
        "num_dependencies": int(len(dep_df)),
        "num_synergy": int((dep_df["type"] == "synergy").sum()),
        "num_redundancy": int((dep_df["type"] == "redundancy").sum()),
        "corr_dep_abs_vs_rule_mean_abs": float(
            dep_df["abs_effective_trained"].corr(dep_df["rule_mean_abs_final_trained"], method="spearman")
        ),
        "corr_dep_abs_vs_rule_max_abs": float(
            dep_df["abs_effective_trained"].corr(dep_df["rule_max_abs_final_trained"], method="spearman")
        ),
        "corr_dep_abs_vs_rule_min_abs": float(
            dep_df["abs_effective_trained"].corr(dep_df["rule_min_abs_final_trained"], method="spearman")
        ),
    }
    for th in NEAR_ZERO_THRESHOLDS:
        summary[f"dep_abs_le_{th:g}"] = float((dep_df["abs_effective_trained"] <= th).mean())
        summary[f"dep_abs_le_{th:g}_given_any_rule_abs_le_1e-3"] = float(
            (dep_df.loc[dep_df["any_rule_near_zero_1e3"], "abs_effective_trained"] <= th).mean()
        )
    semantic = (
        dep_df.groupby("type", as_index=False)
        .agg(
            count=("type", "size"),
            semantic_violation_rate=("semantic_violation", "mean"),
            effective_trained_mean=("effective_trained", "mean"),
            effective_trained_median=("effective_trained", "median"),
        )
        .sort_values("type")
    )
    summary["semantic_by_type"] = semantic.to_dict(orient="records")
    return summary


def build_relation_summary(metrics_df: pd.DataFrame, rules_df: pd.DataFrame, dep_df: pd.DataFrame) -> pd.DataFrame:
    rule_rel = (
        rules_df.groupby("relation", as_index=False)
        .agg(
            num_rules=("ruleID", "size"),
            rule_abs_trained_median=("abs_trained", "median"),
            rule_abs_trained_mean=("abs_trained", "mean"),
            rule_near_zero_rate_1e3=("abs_trained", lambda s: float((s <= 1e-3).mean())),
            rule_near_zero_rate_1e2=("abs_trained", lambda s: float((s <= 1e-2).mean())),
        )
    )
    dep_rel = (
        dep_df.groupby("relation", as_index=False)
        .agg(
            num_dependencies_csv=("type", "size"),
            dep_abs_trained_median=("abs_effective_trained", "median"),
            dep_abs_trained_mean=("abs_effective_trained", "mean"),
            dep_near_zero_rate_1e3=("abs_effective_trained", lambda s: float((s <= 1e-3).mean())),
            semantic_violation_rate=("semantic_violation", "mean"),
        )
    )
    out = metrics_df.merge(rule_rel, on="relation", how="left").merge(dep_rel, on="relation", how="left")
    return out.sort_values(["dependency_stage_accepted", "relation"], ascending=[False, True])


def make_rule_plots(rules_df: pd.DataFrame, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    bins = np.linspace(0, max(rules_df["original"].max(), rules_df["trained"].max()), 80)
    axes[0].hist(rules_df["original"], bins=bins, alpha=0.65, label="original", color="#c57b57")
    axes[0].hist(rules_df["trained"], bins=bins, alpha=0.65, label="trained", color="#4c7aaf")
    axes[0].set_title("Rule Weight Distribution")
    axes[0].set_xlabel("weight")
    axes[0].set_ylabel("count")
    axes[0].legend()

    axes[1].scatter(rules_df["original"], rules_df["trained"], s=6, alpha=0.25, color="#355c7d", linewidths=0)
    max_val = max(rules_df["original"].max(), rules_df["trained"].max())
    axes[1].plot([0, max_val], [0, max_val], linestyle="--", color="#888888", linewidth=1)
    axes[1].set_title("Rule: Original vs Trained")
    axes[1].set_xlabel("original")
    axes[1].set_ylabel("trained")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rule_weight_overview.png"), dpi=180)
    plt.close(fig)


def make_dependency_plots(dep_df: pd.DataFrame, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for dep_type, color in [("synergy", "#2a9d8f"), ("redundancy", "#e76f51")]:
        part = dep_df[dep_df["type"] == dep_type]
        axes[0].hist(
            part["effective_trained"],
            bins=80,
            alpha=0.55,
            label=f"{dep_type} (n={len(part)})",
            color=color,
        )
    axes[0].axvline(0, color="#666666", linestyle="--", linewidth=1)
    axes[0].set_title("Dependency Effective Trained Weight")
    axes[0].set_xlabel("effective_trained")
    axes[0].set_ylabel("count")
    axes[0].legend()

    accepted = dep_df[dep_df["dependency_stage_accepted"] == True]
    rejected = dep_df[dep_df["dependency_stage_accepted"] == False]
    axes[1].scatter(
        accepted["rule_mean_abs_final_trained"],
        accepted["abs_effective_trained"],
        s=6,
        alpha=0.25,
        color="#2a9d8f",
        label=f"accepted (n={len(accepted)})",
        linewidths=0,
    )
    axes[1].scatter(
        rejected["rule_mean_abs_final_trained"],
        rejected["abs_effective_trained"],
        s=6,
        alpha=0.15,
        color="#b0b0b0",
        label=f"rejected (n={len(rejected)})",
        linewidths=0,
    )
    axes[1].set_title("Dependency Magnitude vs Endpoint Rule Magnitude")
    axes[1].set_xlabel("mean abs(final rule weight of the pair)")
    axes[1].set_ylabel("abs(effective dependency weight)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "dependency_weight_overview.png"), dpi=180)
    plt.close(fig)


def write_markdown_report(exp: ExperimentData, out_dir: str) -> str:
    rules_summary = summarize_rules(exp.rules_final)
    dep_summary = summarize_dependencies(exp.dependencies)
    metrics = exp.metrics

    accepted_rel = metrics[metrics["dependency_stage_accepted"] == True]
    rejected_rel = metrics[metrics["dependency_stage_accepted"] == False]
    accepted_dep = exp.dependencies[exp.dependencies["dependency_stage_accepted"] == True]
    rejected_dep = exp.dependencies[exp.dependencies["dependency_stage_accepted"] == False]

    lines = []
    lines.append("# LinearAggregator codex-m analysis")
    lines.append("")
    lines.append("## Experiment scope")
    lines.append(f"- Base dir: `{exp.base_dir}`")
    lines.append(f"- Relations with metrics: {len(metrics)}")
    lines.append(f"- Dependency stage attempted: {int(metrics['dependency_stage_attempted'].sum())}")
    lines.append(f"- Dependency stage accepted: {int((metrics['dependency_stage_accepted'] == True).sum())}")
    lines.append(f"- Dependency stage rejected: {int((metrics['dependency_stage_accepted'] == False).sum())}")
    lines.append("")
    lines.append("## 1. Rule weight shrinkage")
    lines.append(
        f"- Rules total: {rules_summary['num_rules']}; original median={rules_summary['original_median']:.6f}, trained median={rules_summary['trained_median']:.6f}"
    )
    lines.append(
        f"- Fraction with |trained|<=1e-3: {rules_summary['trained_abs_le_0.001']:.2%}; |trained|<=1e-2: {rules_summary['trained_abs_le_0.01']:.2%}"
    )
    lines.append(
        f"- Among rules with original>=0.05, fraction shrunk to |trained|<=1e-3: {rules_summary.get('original_ge_0.05_and_trained_abs_le_1e-3', float('nan')):.2%}"
    )
    lines.append(
        f"- Spearman corr(original, trained)={rules_summary['corr_original_trained']:.3f}; corr(|original|,|trained|)={rules_summary['corr_abs_original_abs_trained']:.3f}"
    )
    lines.append("- Interpretation: base contribution of a near-zero rule is almost removed, but the rule can still matter as a dependency trigger if the dependency stage was accepted for that relation.")
    lines.append("")
    lines.append("## 2. Dependency weight vs rule weight")
    lines.append(
        f"- Dependencies total: {dep_summary['num_dependencies']}; synergy={dep_summary['num_synergy']}, redundancy={dep_summary['num_redundancy']}"
    )
    lines.append(
        f"- Spearman corr(|dep|, mean|rule|)={dep_summary['corr_dep_abs_vs_rule_mean_abs']:.3f}; corr(|dep|, min|rule|)={dep_summary['corr_dep_abs_vs_rule_min_abs']:.3f}; corr(|dep|, max|rule|)={dep_summary['corr_dep_abs_vs_rule_max_abs']:.3f}"
    )
    if not accepted_dep.empty:
        strong_dep_when_any_rule_zero = (accepted_dep.loc[accepted_dep["any_rule_near_zero_1e3"], "abs_effective_trained"] > 1e-2).mean()
        lines.append(
            f"- In accepted dependency-stage relations, when either endpoint rule has |final weight|<=1e-3, the dependency still has |weight|>1e-2 in {strong_dep_when_any_rule_zero:.2%} of pairs."
        )
    lines.append(
        "- Important caveat: `dependency-*.csv` always comes from the dependency-stage model, but `weight-*.csv` comes from the selected final model. For rejected relations, directly comparing the two CSVs mixes different checkpoints."
    )
    lines.append("")
    lines.append("## 3. Semantic consistency of synergy/redundancy")
    for row in dep_summary["semantic_by_type"]:
        lines.append(
            f"- {row['type']}: violation_rate={row['semantic_violation_rate']:.2%}, mean={row['effective_trained_mean']:.6f}, median={row['effective_trained_median']:.6f}"
        )
    if not accepted_dep.empty:
        sem_acc = (
            accepted_dep.groupby("type", as_index=False)["semantic_violation"]
            .mean()
            .rename(columns={"semantic_violation": "violation_rate"})
        )
        for _, row in sem_acc.iterrows():
            lines.append(f"- Accepted-only {row['type']} violation_rate={row['violation_rate']:.2%}")
    if not rejected_dep.empty:
        sem_rej = (
            rejected_dep.groupby("type", as_index=False)["semantic_violation"]
            .mean()
            .rename(columns={"semantic_violation": "violation_rate"})
        )
        for _, row in sem_rej.iterrows():
            lines.append(f"- Rejected-only {row['type']} violation_rate={row['violation_rate']:.2%}")
    lines.append(
        "- Because `sign_constraint_dependency=false` in this experiment, negative synergy and positive redundancy are allowed by the model; they are not bugs in export, but they may indicate the dependency type is acting against its intended semantics."
    )
    lines.append("")
    lines.append("## Optimization ideas")
    lines.append("- If semantic meaning matters, enable dependency sign constraints or add a soft sign regularizer so synergy stays non-negative and redundancy non-positive.")
    lines.append("- Separate reporting for `selected final model` vs `dependency-stage trial model`; otherwise analysis can overstate weird dependency behavior from rejected stage-2 checkpoints.")
    lines.append("- For rules that collapse to zero but still act as dependency triggers, consider decoupling `rule presence` from `rule score` explicitly, or pruning only when both rule weight and all adjacent dependency weights are small.")
    lines.append("- Add structured sparsity: L1/group-lasso on rules and dependencies, ideally with edge-aware pruning so isolated dead rules are removed first.")
    lines.append("- Log real train activation counts in this experiment (`collect_train_hit_counts=true`) before making pruning decisions from weights alone.")

    report = "\n".join(lines) + "\n"
    out_path = os.path.join(out_dir, "report.md")
    with open(out_path, "w") as f:
        f.write(report)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    exp_dir = os.path.abspath(args.experiment_dir)
    out_dir = os.path.abspath(args.output_dir or os.path.join(exp_dir, "analysis_codex"))
    os.makedirs(out_dir, exist_ok=True)

    metrics_df = load_metrics(exp_dir)
    rules_df = load_rule_weights(exp_dir, metrics_df)
    dep_stage_rule_df = load_dependency_stage_rule_weights(exp_dir)
    dep_df = load_dependencies(exp_dir, metrics_df, rules_df)

    exp = ExperimentData(
        base_dir=exp_dir,
        metrics=metrics_df,
        rules_final=rules_df,
        rules_dependency_stage=dep_stage_rule_df,
        dependencies=dep_df,
    )

    relation_summary = build_relation_summary(metrics_df, rules_df, dep_df)
    relation_summary.to_csv(os.path.join(out_dir, "relation_summary.csv"), index=False)
    rules_df.to_csv(os.path.join(out_dir, "rules_merged.csv"), index=False)
    dep_df.to_csv(os.path.join(out_dir, "dependencies_merged.csv"), index=False)

    summary = {
        "rules": summarize_rules(rules_df),
        "dependencies": summarize_dependencies(dep_df),
        "relations": {
            "total": int(len(metrics_df)),
            "dependency_attempted": int(metrics_df["dependency_stage_attempted"].sum()),
            "dependency_accepted": int((metrics_df["dependency_stage_accepted"] == True).sum()),
            "dependency_rejected": int((metrics_df["dependency_stage_accepted"] == False).sum()),
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    make_rule_plots(rules_df, out_dir)
    make_dependency_plots(dep_df, out_dir)
    report_path = write_markdown_report(exp, out_dir)

    print(f"analysis written to: {out_dir}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
