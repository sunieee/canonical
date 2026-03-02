import argparse
import os
import pickle

import torch
from kge import Config, Dataset
from tqdm import tqdm


def save(obj, folder, name):
    if not os.path.exists(folder):
        os.makedirs(folder)
    path_to_file = f"{folder}/{name}.p"
    pickle.dump(obj, open(path_to_file, "wb"))
    return name


def get_dataset(sp_to_o, processed_sp, relation, direction="o"):

    idx_keys_map = dict()
    idx = 0
    for key in sp_to_o.keys():
        if direction == "o":
            e, r = key
        else:
            r, e = key

        if r != relation and relation != -1:
            continue
        if key in processed_sp:
            for ix, candidate in enumerate(processed_sp[key]["candidates"]):
                if len(processed_sp[key]["rules"][ix]) > 0:
                    idx_keys_map[idx] = (e, r, ix)
                    idx += 1
        else:
            idx_keys_map[idx] = None
            idx += 1

    len_ = len(idx_keys_map)
    test = torch.full((len_, 200), PAD_TOK)
    golds = torch.zeros((len_, 1))
    hs = torch.empty((len_, 1))
    rs = torch.empty((len_, 1))
    ts = torch.empty((len_, 1))

    for idx in range(len_):
        (e, r, ix) = idx_keys_map[idx]

        key = (e, r) if direction == "o" else (r, e)

        prediction = processed_sp[key]["candidates"][ix]
        rules = torch.tensor(processed_sp[key]["rules"][ix])
        assert len(rules) <= 200
        assert len(rules) > 0

        if direction == "o":
            hs[idx, 0] = e
            ts[idx, 0] = prediction
        else:
            hs[idx, 0] = prediction
            ts[idx, 0] = e

        rs[idx, 0] = r
        test[idx, : len(rules)] = rules
        golds[idx, 0] = int(prediction in sp_to_o[key])

    return torch.utils.data.TensorDataset(hs, rs, ts, test, golds)


def generate_dataset(relation):

    train_set_o = get_dataset(train_sp_to_o, processed_sp_train, relation)
    train_set_s = get_dataset(train_po_to_s, processed_po_train, relation, direction="s")
    valid_set_o = get_dataset(valid_sp_to_o, processed_sp_valid, relation)
    valid_set_s = get_dataset(valid_po_to_s, processed_po_valid, relation, direction="s")
    test_set_o = get_dataset(test_sp_to_o, processed_sp_test, relation)
    test_set_s = get_dataset(test_po_to_s, processed_po_test, relation, direction="s")

    train_set = torch.utils.data.ConcatDataset([train_set_o, train_set_s])
    test_set = torch.utils.data.ConcatDataset([test_set_o, test_set_s])
    valid_set = torch.utils.data.ConcatDataset([valid_set_o, valid_set_s])

    if args["output"] is not None:
        save((train_set, valid_set, test_set), args["output"], f"dataset_{relation}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Creates datasets for bce")
    parser.add_argument(
        "-e",
        "--explanation",
        help="Folder containing processed explanations",
        default="./codex-m/expl/explanations-processed/",
    )
    parser.add_argument("-d", "--dataset", help="Name of the dataset (loaded with libkge)", default="codex-m")
    parser.add_argument("-o", "--output", help="Folder where datasets are written", default="./codex-m/datasets")
    args = vars(parser.parse_args())

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

    processed_sp_train = pickle.load(open(args["explanation"] + "processed_sp_train.pkl", "rb"))
    processed_po_train = pickle.load(open(args["explanation"] + "processed_po_train.pkl", "rb"))

    processed_sp_test = pickle.load(open(args["explanation"] + "processed_sp_test.pkl", "rb"))
    processed_po_test = pickle.load(open(args["explanation"] + "processed_po_test.pkl", "rb"))

    processed_sp_valid = pickle.load(open(args["explanation"] + "processed_sp_valid.pkl", "rb"))
    processed_po_valid = pickle.load(open(args["explanation"] + "processed_po_valid.pkl", "rb"))

    rule_map = pickle.load(open(args["explanation"] + "rule_map.pkl", "rb"))
    rule_features = pickle.load(open(args["explanation"] + "rule_features.pkl", "rb"))
    ruleid2relid = {ruleid: relid for relid in rule_map for ruleid in rule_map[relid]}

    filter_test = set([tuple(x.tolist()) for x in test_torch])
    filter_valid = set([tuple(x.tolist()) for x in valid_torch])

    LEN_RULES = len(rule_features)
    PAD_TOK = LEN_RULES

    for relation in tqdm(range(dataset.num_relations())):
        generate_dataset(relation)
