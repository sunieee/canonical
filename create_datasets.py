import argparse
import os
import pickle
import re
from multiprocessing import Pool, cpu_count

import torch
from kge import Config, Dataset
from tqdm import tqdm


def _split_rule_line(line: str):
    parts = line.rstrip("\n").split("\t")
    if len(parts) >= 4:
        return parts
    return re.split(r"\s+", line.strip(), maxsplit=3)


def parse_rule_file_stats(rule_file: str):
    num_rules = 0
    max_rule_id = 0
    with open(rule_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            parts = _split_rule_line(line)
            if len(parts) < 4:
                continue
            num_rules += 1
            max_rule_id = line_no
    return num_rules, max_rule_id


def save(obj, folder, name):
    if not os.path.exists(folder):
        os.makedirs(folder)
    path_to_file = f"{folder}/{name}.p"
    pickle.dump(obj, open(path_to_file, "wb"))
    return name


def read_ids(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read().splitlines()
    return [line.split("\t")[1] for line in raw]


def load_applied_rules(path):
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_processed_from_applied(applied_rules, entity_id_to_idx, relation_id_to_idx):
    processed_sp = {}
    processed_po = {}

    tail_applied = applied_rules.get("tail", {})
    for rel_raw, source_map in tail_applied.items():
        if rel_raw not in relation_id_to_idx:
            continue
        p_idx = relation_id_to_idx[rel_raw]
        for s_raw, target_map in source_map.items():
            if s_raw not in entity_id_to_idx:
                continue
            s_idx = entity_id_to_idx[s_raw]
            key = (s_idx, p_idx)
            bucket = processed_sp.setdefault(key, {"candidates": [], "rules": []})

            for o_raw, rule_ids in target_map.items():
                if o_raw not in entity_id_to_idx:
                    continue
                o_idx = entity_id_to_idx[o_raw]
                ids = [int(rid) for rid in rule_ids if int(rid) > 0]
                bucket["candidates"].append(o_idx)
                bucket["rules"].append(ids)

    head_applied = applied_rules.get("head", {})
    for rel_raw, source_map in head_applied.items():
        if rel_raw not in relation_id_to_idx:
            continue
        p_idx = relation_id_to_idx[rel_raw]
        for o_raw, target_map in source_map.items():
            if o_raw not in entity_id_to_idx:
                continue
            o_idx = entity_id_to_idx[o_raw]
            key = (p_idx, o_idx)
            bucket = processed_po.setdefault(key, {"candidates": [], "rules": []})

            for s_raw, rule_ids in target_map.items():
                if s_raw not in entity_id_to_idx:
                    continue
                s_idx = entity_id_to_idx[s_raw]
                ids = [int(rid) for rid in rule_ids if int(rid) > 0]
                bucket["candidates"].append(s_idx)
                bucket["rules"].append(ids)

    return processed_sp, processed_po


def build_compact_split(sp_to_o, processed_sp, relation, direction="o"):
    rules_flat = []
    offsets = [0]
    golds = []

    for key in sp_to_o.keys():
        if direction == "o":
            e, r = key
        else:
            r, e = key

        if r != relation and relation != -1:
            continue
        if key not in processed_sp:
            continue

        candidates = processed_sp[key]["candidates"]
        rules_per_candidate = processed_sp[key]["rules"]
        for ix, prediction in enumerate(candidates):
            rule_ids = rules_per_candidate[ix]
            if len(rule_ids) == 0:
                continue
            rules_flat.extend(rule_ids)
            offsets.append(len(rules_flat))
            golds.append(int(prediction in sp_to_o[key]))

    rules_flat_t = torch.tensor(rules_flat, dtype=torch.int32)
    offsets_t = torch.tensor(offsets, dtype=torch.int64)
    golds_t = torch.tensor(golds, dtype=torch.float32).reshape(-1, 1)

    return {
        "rules_flat": rules_flat_t,
        "offsets": offsets_t,
        "golds": golds_t,
        "num_samples": int(golds_t.shape[0]),
    }


def concat_compact_splits(split_a, split_b):
    if split_a["num_samples"] == 0:
        return split_b
    if split_b["num_samples"] == 0:
        return split_a

    rules_flat = torch.cat([split_a["rules_flat"], split_b["rules_flat"]], dim=0)
    offsets_b_shifted = split_b["offsets"][1:] + split_a["rules_flat"].shape[0]
    offsets = torch.cat([split_a["offsets"], offsets_b_shifted], dim=0)
    golds = torch.cat([split_a["golds"], split_b["golds"]], dim=0)

    return {
        "rules_flat": rules_flat,
        "offsets": offsets,
        "golds": golds,
        "num_samples": int(golds.shape[0]),
    }


def generate_dataset(relation):
    train_set_o = build_compact_split(train_sp_to_o, processed_sp_train, relation)
    train_set_s = build_compact_split(train_po_to_s, processed_po_train, relation, direction="s")

    train_set = concat_compact_splits(train_set_o, train_set_s)

    data_obj = {
        "format": "compact_varlen_int32_v1",
        "pad_tok": int(PAD_TOK),
        "num_rules": int(LEN_RULES),
        "train": train_set,
    }

    if args["output"] is not None:
        save(data_obj, args["output"], f"dataset_{relation}")

    return relation


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Creates datasets for bce")
    parser.add_argument("-d", "--dataset", help="Name of the dataset (loaded with libkge)", default="codex-m")
    parser.add_argument("--data_root", help="Dataset root folder", default="data")
    parser.add_argument(
        "--applied_rules",
        help="Path to applied_rules_train.json",
        default=None,
    )
    parser.add_argument("--rule_file", help="Path to rules file", default="")
    parser.add_argument("-o", "--output", help="Folder where datasets are written", default=None)
    args = vars(parser.parse_args())
    dataset_dir = os.path.join(args["data_root"], args["dataset"])
    if args["applied_rules"] is None:
        args["applied_rules"] = os.path.join(dataset_dir, "expl", "applied_rules_train.json")
    if args["output"] is None:
        args["output"] = os.path.join(dataset_dir, "datasets")
    if args["rule_file"] == "":
        args["rule_file"] = os.path.join(dataset_dir, "rules", "rules-1000")

    c = Config()
    c.set("dataset.name", args["dataset"])
    dataset = Dataset.create(c)

    if not os.path.exists(args["output"]):
        os.makedirs(args["output"])

    train_sp_to_o = dataset.index("train_sp_to_o")
    train_po_to_s = dataset.index("train_po_to_s")

    entity_ids = read_ids(os.path.join(dataset_dir, "entity_ids.del"))
    relation_ids = read_ids(os.path.join(dataset_dir, "relation_ids.del"))
    entity_id_to_idx = {ent: idx for idx, ent in enumerate(entity_ids)}
    relation_id_to_idx = {rel: idx for idx, rel in enumerate(relation_ids)}

    applied_rules_train = load_applied_rules(args["applied_rules"])
    processed_sp_train, processed_po_train = build_processed_from_applied(
        applied_rules_train,
        entity_id_to_idx,
        relation_id_to_idx,
    )

    LEN_RULES, MAX_RULE_ID = parse_rule_file_stats(args["rule_file"])
    PAD_TOK = MAX_RULE_ID + 1

    num_relations = dataset.num_relations()
    num_workers = cpu_count()

    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(generate_dataset, range(num_relations)), total=num_relations))

    # for relation in tqdm(range(dataset.num_relations())):
    #     generate_dataset(relation)
