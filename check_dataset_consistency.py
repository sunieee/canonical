#!/usr/bin/env python3
import argparse
import os
import pickle
import re
from pathlib import Path

import torch


REL_FILE_PATTERN = re.compile(r"^dataset_(-?\d+)\.p$")


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def list_relation_files(folder: Path):
    rel_to_file = {}
    for name in os.listdir(folder):
        m = REL_FILE_PATTERN.match(name)
        if m:
            rel_to_file[int(m.group(1))] = folder / name
    return rel_to_file


def old_split_to_compact(split_obj, pad_tok: int):
    """
    Convert old-format split (TensorDataset or ConcatDataset of TensorDataset)
    to compact format: rules_flat, offsets, golds, num_samples.
    """
    if isinstance(split_obj, torch.utils.data.ConcatDataset):
        datasets = split_obj.datasets
    else:
        datasets = [split_obj]

    rules_parts = []
    gold_parts = []

    for ds in datasets:
        if not isinstance(ds, torch.utils.data.TensorDataset):
            raise TypeError(f"Unexpected dataset type in old split: {type(ds)}")
        if len(ds.tensors) != 5:
            raise ValueError(
                f"Expected 5 tensors (h, r, t, rules, gold), got {len(ds.tensors)}"
            )

        _, _, _, rules_padded, golds = ds.tensors
        rules_parts.append(rules_padded)
        gold_parts.append(golds)

    if len(rules_parts) == 0:
        rules_padded_all = torch.empty((0, 0), dtype=torch.int64)
        golds_all = torch.empty((0, 1), dtype=torch.float32)
    else:
        rules_padded_all = torch.cat(rules_parts, dim=0)
        golds_all = torch.cat(gold_parts, dim=0)

    rules_flat = []
    offsets = [0]
    for row in rules_padded_all:
        rule_ids = row[row != pad_tok].to(torch.int64)
        rules_flat.extend(rule_ids.tolist())
        offsets.append(len(rules_flat))

    return {
        "rules_flat": torch.tensor(rules_flat, dtype=torch.int64),
        "offsets": torch.tensor(offsets, dtype=torch.int64),
        "golds": golds_all.reshape(-1, 1).to(torch.float32),
        "num_samples": int(golds_all.shape[0]),
    }


def infer_old_pad_tok(old_obj):
    """
    Infer pad token used by old padded format.
    In old datasets, pad token is LEN_RULES and all valid rule ids are < LEN_RULES,
    so the maximum value in padded rule tensor is expected to be PAD_TOK.
    """
    max_vals = []
    for split_obj in old_obj:
        if isinstance(split_obj, torch.utils.data.ConcatDataset):
            datasets = split_obj.datasets
        else:
            datasets = [split_obj]

        for ds in datasets:
            if not isinstance(ds, torch.utils.data.TensorDataset):
                continue
            if len(ds.tensors) != 5:
                continue
            rules_padded = ds.tensors[3]
            if rules_padded.numel() > 0:
                max_vals.append(int(rules_padded.max().item()))

    if not max_vals:
        raise ValueError("Cannot infer old pad token from old dataset")

    return max(max_vals)


def normalize_new_split(split_obj):
    required = {"rules_flat", "offsets", "golds", "num_samples"}
    if not isinstance(split_obj, dict) or not required.issubset(set(split_obj.keys())):
        raise TypeError("New split is not in expected compact dict format")

    return {
        "rules_flat": split_obj["rules_flat"].to(torch.int64),
        "offsets": split_obj["offsets"].to(torch.int64),
        "golds": split_obj["golds"].reshape(-1, 1).to(torch.float32),
        "num_samples": int(split_obj["num_samples"]),
    }


def compare_compact(old_c, new_c, split_name, relation, max_show=3):
    errors = []

    n_old = old_c["num_samples"]
    n_new = new_c["num_samples"]
    if n_old != n_new:
        errors.append(
            f"relation={relation} split={split_name}: num_samples mismatch old={n_old}, new={n_new}"
        )

    if old_c["offsets"].shape != new_c["offsets"].shape or not torch.equal(
        old_c["offsets"], new_c["offsets"]
    ):
        errors.append(
            f"relation={relation} split={split_name}: offsets mismatch "
            f"old_shape={tuple(old_c['offsets'].shape)}, new_shape={tuple(new_c['offsets'].shape)}"
        )

    if old_c["rules_flat"].shape != new_c["rules_flat"].shape or not torch.equal(
        old_c["rules_flat"], new_c["rules_flat"]
    ):
        errors.append(
            f"relation={relation} split={split_name}: rules_flat mismatch "
            f"old_len={old_c['rules_flat'].numel()}, new_len={new_c['rules_flat'].numel()}"
        )

    if old_c["golds"].shape != new_c["golds"].shape or not torch.equal(
        old_c["golds"], new_c["golds"]
    ):
        errors.append(
            f"relation={relation} split={split_name}: golds mismatch "
            f"old_shape={tuple(old_c['golds'].shape)}, new_shape={tuple(new_c['golds'].shape)}"
        )

    # If there are mismatches, provide sample-level details (first few only)
    if errors:
        detailed = []
        limit = min(n_old, n_new)
        shown = 0
        old_offsets = old_c["offsets"]
        new_offsets = new_c["offsets"]

        for i in range(limit):
            old_s, old_e = int(old_offsets[i].item()), int(old_offsets[i + 1].item())
            new_s, new_e = int(new_offsets[i].item()), int(new_offsets[i + 1].item())

            old_rules = old_c["rules_flat"][old_s:old_e]
            new_rules = new_c["rules_flat"][new_s:new_e]

            old_gold = float(old_c["golds"][i].item())
            new_gold = float(new_c["golds"][i].item())

            if (not torch.equal(old_rules, new_rules)) or (old_gold != new_gold):
                detailed.append(
                    f"  sample_idx={i}: old_rules={old_rules.tolist()}, new_rules={new_rules.tolist()}, "
                    f"old_gold={old_gold}, new_gold={new_gold}"
                )
                shown += 1
                if shown >= max_show:
                    break

        if detailed:
            errors.extend(detailed)

    return errors


def compare_relation_file(old_file: Path, new_file: Path, max_show=3):
    relation = int(REL_FILE_PATTERN.match(new_file.name).group(1))

    old_obj = load_pickle(old_file)
    new_obj = load_pickle(new_file)

    # old format: (train_set, valid_set, test_set)
    if not (isinstance(old_obj, tuple) and len(old_obj) == 3):
        raise TypeError(
            f"Old file {old_file} is not expected tuple(train, valid, test), got {type(old_obj)}"
        )

    # new format: dict with metadata + train/valid/test compact splits
    if not isinstance(new_obj, dict):
        raise TypeError(f"New file {new_file} is not expected dict format")

    if "pad_tok" not in new_obj:
        raise KeyError(f"New file {new_file} missing key 'pad_tok'")

    pad_tok_new = int(new_obj["pad_tok"])
    pad_tok_old = infer_old_pad_tok(old_obj)

    split_map_old = {
        "train": old_obj[0],
        "valid": old_obj[1],
        "test": old_obj[2],
    }

    all_errors = []
    for split_name in ["train", "valid", "test"]:
        if split_name not in new_obj:
            all_errors.append(
                f"relation={relation}: new file missing split '{split_name}'"
            )
            continue

        old_compact = old_split_to_compact(split_map_old[split_name], pad_tok=pad_tok_old)
        new_compact = normalize_new_split(new_obj[split_name])
        errs = compare_compact(old_compact, new_compact, split_name, relation, max_show=max_show)
        all_errors.extend(errs)

    if pad_tok_old != pad_tok_new:
        all_errors.append(
            f"relation={relation}: pad_tok differs old={pad_tok_old}, new={pad_tok_new}"
        )

    return all_errors


def main():
    parser = argparse.ArgumentParser(
        description="Compare old-format datasets.bac and new-format datasets for consistency"
    )
    parser.add_argument(
        "--new-dir",
        default="./fb15k-237/datasets",
        help="Path to new-format datasets directory",
    )
    parser.add_argument(
        "--old-dir",
        default="./fb15k-237/datasets.bac",
        help="Path to old-format datasets directory",
    )
    parser.add_argument(
        "--relation",
        type=int,
        default=None,
        help="Only check one relation id (e.g. 17). If omitted, check all common relation files.",
    )
    parser.add_argument(
        "--max-show",
        type=int,
        default=3,
        help="Max mismatched sample details to print per split",
    )

    args = parser.parse_args()

    new_dir = Path(args.new_dir)
    old_dir = Path(args.old_dir)

    if not new_dir.exists():
        raise FileNotFoundError(f"new-dir not found: {new_dir}")
    if not old_dir.exists():
        raise FileNotFoundError(f"old-dir not found: {old_dir}")

    new_files = list_relation_files(new_dir)
    old_files = list_relation_files(old_dir)

    if args.relation is not None:
        rels = [args.relation]
    else:
        rels = sorted(set(new_files.keys()) & set(old_files.keys()))

    missing_in_new = sorted(set(old_files.keys()) - set(new_files.keys()))
    missing_in_old = sorted(set(new_files.keys()) - set(old_files.keys()))

    if missing_in_new:
        print(f"[WARN] Missing in new-dir: {missing_in_new[:20]}" + (" ..." if len(missing_in_new) > 20 else ""))
    if missing_in_old:
        print(f"[WARN] Missing in old-dir: {missing_in_old[:20]}" + (" ..." if len(missing_in_old) > 20 else ""))

    if len(rels) == 0:
        print("No common relation files found to compare.")
        raise SystemExit(2)

    total = 0
    failed = 0

    for rel in rels:
        total += 1
        if rel not in old_files:
            print(f"[FAIL] relation={rel}: missing in old-dir")
            failed += 1
            continue
        if rel not in new_files:
            print(f"[FAIL] relation={rel}: missing in new-dir")
            failed += 1
            continue

        try:
            errs = compare_relation_file(old_files[rel], new_files[rel], max_show=args.max_show)
        except Exception as e:
            print(f"[FAIL] relation={rel}: exception={e}")
            failed += 1
            continue

        if errs:
            print(f"[FAIL] relation={rel}")
            for e in errs:
                print("   " + e)
            failed += 1
        else:
            print(f"[OK] relation={rel}")

    print("\n===== SUMMARY =====")
    print(f"checked: {total}")
    print(f"failed : {failed}")
    print(f"passed : {total - failed}")

    raise SystemExit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
