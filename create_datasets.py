import argparse
import os
import pickle
from multiprocessing import Pool, cpu_count

import torch
from kge import Config, Dataset
from tqdm import tqdm


def save(obj, folder, name):
    if not os.path.exists(folder):
        os.makedirs(folder)
    path_to_file = f"{folder}/{name}.p"
    pickle.dump(obj, open(path_to_file, "wb"))
    return name


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
    valid_set_o = build_compact_split(valid_sp_to_o, processed_sp_valid, relation)
    valid_set_s = build_compact_split(valid_po_to_s, processed_po_valid, relation, direction="s")
    test_set_o = build_compact_split(test_sp_to_o, processed_sp_test, relation)
    test_set_s = build_compact_split(test_po_to_s, processed_po_test, relation, direction="s")

    train_set = concat_compact_splits(train_set_o, train_set_s)
    valid_set = concat_compact_splits(valid_set_o, valid_set_s)
    test_set = concat_compact_splits(test_set_o, test_set_s)

    data_obj = {
        "format": "compact_varlen_int32_v1",
        "pad_tok": int(PAD_TOK),
        "num_rules": int(LEN_RULES),
        "train": train_set,
        "valid": valid_set,
        "test": test_set,
    }

    if args["output"] is not None:
        save(data_obj, args["output"], f"dataset_{relation}")

    return relation


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Creates datasets for bce")
    # parser.add_argument("-e", "--explanation", help="Folder containing processed explanations", default=None)
    parser.add_argument("-d", "--dataset", help="Name of the dataset (loaded with libkge)", default="codex-m")
    # parser.add_argument("-o", "--output", help="Folder where datasets are written", default="./codex-m/datasets")
    args = vars(parser.parse_args())
    args["explanation"] = os.path.join(args["dataset"], "expl", "explanations-processed")
    args["output"] = os.path.join(args["dataset"], "datasets")

    c = Config()
    c.set("dataset.name", args["dataset"])
    dataset = Dataset.create(c)

    if not os.path.exists(args["output"]):
        os.makedirs(args["output"])

    train_sp_to_o = dataset.index("train_sp_to_o")
    train_po_to_s = dataset.index("train_po_to_s")

    test_sp_to_o = dataset.index("test_sp_to_o")
    test_po_to_s = dataset.index("test_po_to_s")
    test_torch = dataset.split("test")

    valid_sp_to_o = dataset.index("valid_sp_to_o")
    valid_po_to_s = dataset.index("valid_po_to_s")
    valid_torch = dataset.split("valid")

    processed_sp_train = pickle.load(open(os.path.join(args["explanation"], "processed_sp_train.pkl"), "rb"))
    processed_po_train = pickle.load(open(os.path.join(args["explanation"], "processed_po_train.pkl"), "rb"))

    processed_sp_test = pickle.load(open(os.path.join(args["explanation"], "processed_sp_test.pkl"), "rb"))
    processed_po_test = pickle.load(open(os.path.join(args["explanation"], "processed_po_test.pkl"), "rb"))

    processed_sp_valid = pickle.load(open(os.path.join(args["explanation"], "processed_sp_valid.pkl"), "rb"))
    processed_po_valid = pickle.load(open(os.path.join(args["explanation"], "processed_po_valid.pkl"), "rb"))

    rule_map = pickle.load(open(os.path.join(args["explanation"], "rule_map.pkl"), "rb"))
    rule_features = pickle.load(open(os.path.join(args["explanation"], "rule_features.pkl"), "rb"))
    ruleid2relid = {ruleid: relid for relid in rule_map for ruleid in rule_map[relid]}

    filter_test = set([tuple(x.tolist()) for x in test_torch])
    filter_valid = set([tuple(x.tolist()) for x in valid_torch])

    LEN_RULES = len(rule_features)
    PAD_TOK = LEN_RULES

    num_relations = dataset.num_relations()
    num_workers = cpu_count()

    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(generate_dataset, range(num_relations)), total=num_relations))

    # for relation in tqdm(range(dataset.num_relations())):
    #     generate_dataset(relation)
