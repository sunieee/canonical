#!/usr/bin/env python
# coding: utf-8
import argparse
import copy
import ctypes
import itertools
import json
import logging
import math
import os
import pickle
import shutil
import uuid
import warnings
from datetime import datetime
from os.path import exists
from pprint import pformat

import kge
import numpy as np
import scipy
import torch
from torch import multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")


def save(obj, folder, name=None, override=False):
    if name is None:
        name = uuid.uuid4()
    if not os.path.exists(folder):
        os.makedirs(folder)
    path_to_file = f"{folder}/{name}.p"
    if exists(path_to_file):
        print(f"Warning name {name} exists in cache, do you want to overwrite y/n?")
        confirm = input() if not override else "y"
        if confirm != "y":
            return None

    pickle.dump(obj, open(path_to_file, "wb"))
    return name


def load(folder, name):
    path_to_file = f"{folder}/{name}.p"
    if exists(path_to_file):
        return pickle.load(open(f"{folder}/{name}.p", "rb"))
    else:
        print("No such name in cache")
        return None


def train(dataloader, model, loss_fn, optimizer, reg=False, num_unseen=0):
    model.train()
    train_loss = 0
    n_loss = 0
    for i, (rules, y) in enumerate(dataloader):

        # compute regularization
        if reg and num_unseen > 0:
            num_batches = len(dataloader)
            if num_unseen > num_batches:
                num_unseen = num_batches
            if i % int(num_batches / num_unseen) == 0:
                rule_confs = torch.nn.functional.sigmoid(model.rules.weight)
                sudo_false = torch.zeros_like(rule_confs)
                loss = loss_fn(rule_confs, sudo_false) / dataloader.batch_size
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Compute prediction error

        rules = rules.long().to(args.device)
        y = y.to(args.device)
        pred = model(rules)
        loss = loss_fn(pred.reshape(-1, 1), y)

        train_loss += loss.item()
        n_loss += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss / n_loss


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for i, (rules, y) in enumerate(dataloader):

            rules = rules.long().to(args.device)
            y = y.to(args.device)

            pred = model(rules).reshape(-1, 1)

            loss = loss_fn(pred, y)
            test_loss += loss.item()

            correct += ((torch.sigmoid(pred) > 0.5) == y.to(args.device)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    logging.info(f"Test Error: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return test_loss


def get_conf(x):
    if x == PAD_TOK:
        return 0.0
    rule = rule_features[x]
    return int(rule[1]) / (int(rule[0]) + 5)


get_conf = np.vectorize(get_conf, otypes=[float])


def rank_batch(nnm, golds, candidates, rules, test_filter):

    batch_rank, batch_rank_raw = [], []
    if len(candidates) > 0 and len(rules) > 0:
        fill_value = 0.0
        scores = torch.full((dataset.num_entities(),), fill_value)
        scores_raw = torch.full((dataset.num_entities(),), fill_value)

        rules_ = torch.nested.to_padded_tensor(
            torch.nested.nested_tensor([torch.tensor(x) for x in rules]), padding=PAD_TOK
        ).long()
        if rules_.numel() == 0:
            return torch.tensor(batch_rank), torch.tensor(batch_rank_raw), len(golds)

        with torch.no_grad():
            pred = nnm(rules_).detach()
            if args.model != "NoisyOrAggregator":
                pred = torch.sigmoid(pred).detach()

        max_conf = get_conf(rules_.cpu()).max(axis=1).reshape(-1, 1).astype(np.float32)

        scores[candidates] = (pred * max_conf).squeeze(dim=1)
        scores_raw[candidates] = pred.squeeze(dim=1)

        def get_rank(scores):
            batch_rank = []
            scores = -1 * scores
            gold_scores = scores[golds].clone()
            scores[golds] = fill_value
            if test_filter is not None:
                scores[test_filter] = fill_value
            for ix, gold in enumerate(golds):
                gold = gold.item()
                scores[gold] = gold_scores[ix]
                ranking = scipy.stats.rankdata(scores.detach().numpy())  # .detach().numpy()
                batch_rank.append(ranking[gold])
                scores[gold] = fill_value
            return batch_rank

        batch_rank = get_rank(scores)
        batch_rank_raw = get_rank(scores_raw)

    return torch.tensor(batch_rank), torch.tensor(batch_rank_raw), len(golds)


def get_ranks(nnm, sp_to_o, processed, relation, direction="o", filter_test=False):
    nnm.eval()
    if direction == "o":
        # sp
        keys = [key for key in sp_to_o.keys() if key[1] == relation]
    else:
        # po
        keys = [key for key in sp_to_o.keys() if key[0] == relation]

    if len(keys) == 0:
        return torch.tensor([]), torch.tensor([]), 0

    data = []
    for key in keys:
        test_filter = None
        if filter_test:
            if direction == "o":
                if key in test_sp_to_o.keys():
                    test_filter = test_sp_to_o[key].long()
            else:
                if key in test_po_to_s.keys():
                    test_filter = test_po_to_s[key].long()

        golds = sp_to_o[key].long()
        candidates = []
        rules = []
        if key in processed:
            candidates = processed[key]["candidates"]
            rules = processed[key]["rules"]
        data.append((nnm, golds, candidates, rules, test_filter))

    data = itertools.starmap(rank_batch, data)
    rank, rank_raw, ns = zip(*data)
    return torch.hstack(rank), torch.hstack(rank_raw), sum(ns)


class LinearAggregator(nn.Module):
    def init_weights(self):
        with torch.no_grad():
            torch.manual_seed(0)
            confs = torch.from_numpy(get_conf(self.relation_rule_ids)).reshape(-1, 1).float()
            self.rules.weight[: self.num_relation_rules] = confs
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.rules.weight[: self.num_relation_rules].reshape(1, -1))
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            self.bias.uniform_(-bound, bound)

    def __init__(self, relation, sign_constraint=False):
        super().__init__()
        self.sign_constraint = sign_constraint

        relation_rule_ids = sorted(rule_map.get(relation, []))
        self.relation_rule_ids = np.array(relation_rule_ids, dtype=np.int64)
        self.num_relation_rules = len(relation_rule_ids)
        self.pad_local_tok = self.num_relation_rules

        self.rules = nn.Embedding(self.num_relation_rules + 1, 1, padding_idx=self.pad_local_tok)
        self.bias = nn.Parameter(torch.zeros(1, 1))

        global_to_local = torch.full((LEN_RULES + 1,), self.pad_local_tok, dtype=torch.long)
        if self.num_relation_rules > 0:
            global_to_local[torch.tensor(relation_rule_ids, dtype=torch.long)] = torch.arange(
                self.num_relation_rules, dtype=torch.long
            )
        self.register_buffer("global_to_local", global_to_local)

        self.init_weights()

    def forward(self, rules):
        local_rules = self.global_to_local[rules.long()]
        mask = local_rules == self.pad_local_tok
        local_rules = self.rules(local_rules)
        local_rules.masked_fill_(mask.unsqueeze(dim=2), 0.0)
        if self.sign_constraint:
            local_rules = local_rules**2
        logits = local_rules.sum(dim=1) + self.bias
        return logits


class NoisyOrAggregator(nn.Module):
    def init_weights(self):
        with torch.no_grad():
            torch.manual_seed(0)
            confs = torch.from_numpy(get_conf(self.relation_rule_ids)).reshape(-1, 1).float()
            confs = confs.clamp(min=1e-6, max=1 - 1e-6)
            logit_values = torch.log(confs / (1 - confs))
            self.rules.weight[: self.num_relation_rules] = logit_values.float()

    def __init__(self, relation):
        super().__init__()
        relation_rule_ids = sorted(rule_map.get(relation, []))
        self.relation_rule_ids = np.array(relation_rule_ids, dtype=np.int64)
        self.num_relation_rules = len(relation_rule_ids)
        self.pad_local_tok = self.num_relation_rules

        self.rules = nn.Embedding(self.num_relation_rules + 1, 1, padding_idx=self.pad_local_tok)

        global_to_local = torch.full((LEN_RULES + 1,), self.pad_local_tok, dtype=torch.long)
        if self.num_relation_rules > 0:
            global_to_local[torch.tensor(relation_rule_ids, dtype=torch.long)] = torch.arange(
                self.num_relation_rules, dtype=torch.long
            )
        self.register_buffer("global_to_local", global_to_local)

        self.init_weights()

    def forward(self, rules):
        local_rules = self.global_to_local[rules.long()]
        mask = local_rules == self.pad_local_tok
        local_rules = self.rules(local_rules)
        local_rules.masked_fill_(mask.unsqueeze(dim=2), float("-inf"))
        no = 1 - (1 - torch.nn.functional.sigmoid(local_rules)).prod(dim=1)
        no = no.clamp(min=0.0001, max=0.99999)
        return no


def calc_mrr(tail_mrr, head_mrr, attr="maximums_t"):
    relation = tail_mrr.relation
    if relation != head_mrr.relation:
        raise ValueError("head_mrr and tail_mrr must track the same relation")

    rn = test_torch[test_torch[:, 1] == relation].shape[0]
    if rn == 0:
        return 0.0, 0.0

    tail_rank = getattr(tail_mrr, attr) * rn
    head_rank = getattr(head_mrr, attr) * rn
    tail_rank_raw = getattr(tail_mrr, attr + "_raw") * rn
    head_rank_raw = getattr(head_mrr, attr + "_raw") * rn

    return (head_rank + tail_rank) / (2 * rn), (head_rank_raw + tail_rank_raw) / (2 * rn)


class MRR:
    def __init__(self, relation, direction="o"):
        self.relation = relation
        self.direction = direction

        self.best_hps = None
        self.best_hps_raw = None

        self.maximums_v = 0.0
        self.maximums_v_raw = 0.0

        self.maximums_t = 0.0
        self.maximums_t_raw = 0.0
        self.maximums_t_1 = 0.0
        self.maximums_t_1_raw = 0.0
        self.maximums_t_10 = 0.0
        self.maximums_t_10_raw = 0.0

        self.valid_sp_to_o = valid_sp_to_o if direction == "o" else valid_po_to_s
        self.valid_processed = processed_sp_valid if direction == "o" else processed_po_valid
        self.test_sp_to_o = test_sp_to_o if direction == "o" else test_po_to_s
        self.test_processed = processed_sp_test if direction == "o" else processed_po_test
        self.nnm = None
        self.nnm_raw = None

    def calc_metrics_(self, ranks, n):
        if n == 0:
            return 0.0, 0.0, 0.0
        mrr = ((1 / ranks).sum() / n).item()
        h1 = ((ranks == 1.0).sum() / n).item()
        h10 = ((ranks <= 10.0).sum() / n).item()
        return mrr, h1, h10

    def calc_metrics(self, nnm, sp_to_o, processed, direction, filter_test=False):
        relation = self.relation
        ranks, ranks_raw, n = get_ranks(nnm, sp_to_o, processed, relation, direction, filter_test)
        mrr, h1, h10 = self.calc_metrics_(ranks, n)
        mrr_raw, h1_raw, h10_raw = self.calc_metrics_(ranks_raw, n)
        return (mrr, h1, h10, mrr_raw, h1_raw, h10_raw)

    def update(self, nnm, hps):
        (v_mrr, v_h1, v_h10, v_mrr_raw, v_h1_raw, v_h10_raw) = self.calc_metrics(
            nnm, self.valid_sp_to_o, self.valid_processed, direction=self.direction, filter_test=True
        )
        if (v_mrr > self.maximums_v) or (v_mrr_raw > self.maximums_v_raw):
            (t_mrr, t_h1, t_h10, t_mrr_raw, t_h1_raw, t_h10_raw) = self.calc_metrics(
                nnm, self.test_sp_to_o, self.test_processed, direction=self.direction
            )
            if v_mrr > self.maximums_v:
                self.maximums_v = v_mrr
                self.maximums_t = t_mrr
                self.maximums_t_1 = t_h1
                self.maximums_t_10 = t_h10
                self.nnm = copy.deepcopy(nnm)
                self.best_hps = hps

            if v_mrr_raw > self.maximums_v_raw:
                self.maximums_v_raw = v_mrr_raw
                self.maximums_t_raw = t_mrr_raw
                self.maximums_t_1_raw = t_h1_raw
                self.maximums_t_10_raw = t_h10_raw
                self.nnm_raw = copy.deepcopy(nnm)
                self.best_hps_raw = hps


class SharedDataset(Dataset):
    def get_empty_shared_array(self, shape, type_):
        shared_array_base = mp.Array(type_, torch.tensor(shape).prod().item())
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array = shared_array.reshape(*shape)
        return torch.from_numpy(shared_array)

    def __init__(self, xs, ys):
        self.shared_x = self.get_empty_shared_array(xs.shape, ctypes.c_int)
        self.shared_x[:] = xs
        self.shared_y = self.get_empty_shared_array(ys.shape, ctypes.c_float)
        self.shared_y[:] = ys
        self.use_cache = False
        self.len = xs.shape[0]

    def __getitem__(self, index):
        x = self.shared_x[index]
        y = self.shared_y[index]
        return x, y

    def __len__(self):
        return self.len


def load_dataloaders(dataset_directory, relation):

    train_set, _, _ = load(dataset_directory, f"dataset_{relation}")

    train_set = SharedDataset(
        torch.vstack((train_set.datasets[0].tensors[3], train_set.datasets[1].tensors[3])),
        torch.vstack((train_set.datasets[0].tensors[4], train_set.datasets[1].tensors[4])),
    )
    if len(train_set) == 0:
        return None
    train_dataloader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=args.shuffle_train, num_workers=args.max_worker_dataloader
    )
    return train_dataloader


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", action="store", help="Name of dataset (libkge)", default="codex-m")
    parser.add_argument("-dev", "--device", action="store", help="Device cpu/cuda", default="cuda")
    parser.add_argument(
        "--max_worker_dataloader",
        action="store",
        help="Number of processes for dataloader",
        default=len(os.sched_getaffinity(0)) - 1,
    )
    parser.add_argument(
        "--max_worker_mrr",
        action="store",
        help="Number of processes working on MRR evaluation",
        default=len(os.sched_getaffinity(0)) - 1,
    )
    parser.add_argument(
        "--model",
        action="store",
        help="Aggregator to use; one of ['LinearAggregator', 'NoisyOrAggregator']",
        default="LinearAggregator",
    )
    parser.add_argument("--shuffle_train", action="store_true", help="Shuffles the examples before creating batches")
    parser.add_argument("--batch_size", action="store", help="Size of batch", default=4096)
    parser.add_argument("--lr", action="store", default=0.001, help="Learning rates of the adam optimizer")
    parser.add_argument("--max_epoch", action="store", default=10, help="Epochs to run for each learning rate")
    parser.add_argument("--pos", action="store", default=15, help="Scaling of the loss for positive examples")
    parser.add_argument(
        "--sign_constraint",
        action="store_true",
        help="Constrains the rule weights to be >=0. Only implemented for LinearAggregator.",
    )
    parser.add_argument(
        "--noisy_or_reg", action="store_true", help="Sudo negative examples for noisy-or learning.", default=False
    )
    parser.add_argument(
        "--num_unseen", action="store_true", help="Num Sudo negative examples for noisy-or learning.", default=0
    )
    parser.add_argument("--relation", action="store", help="Relation to train on", default=0, type=int)

    return parser


def BCELossR(weights=[1, 1], reduction="mean", apply_sigmoid=False):
    def loss(input, target):
        if apply_sigmoid:
            input = torch.sigmoid(input)
            input = torch.clamp(input, min=1e-7, max=1 - 1e-7)
        bce = -weights[1] * target * torch.log(input) - (1 - target) * weights[0] * torch.log(1 - input)
        if reduction == "libkge":
            bce = (
                bce[target.bool()].sum() / target.bool().sum() + bce[~target.bool()].sum() / (~target.bool()).sum()
            ) / 2.0
        elif reduction == "sum":
            bce = torch.sum(bce)
        elif reduction == "mean":
            bce = torch.mean(bce)
        return bce

    return loss


def aggregate_single(relation):
    dataloader = load_dataloaders(args.directory_preprocessed_datasets, relation)

    pos = args.pos
    lr = args.lr
    max_epoch = args.max_epoch
    unseen = args.num_unseen
    tail_mrr = MRR(relation=relation, direction="o")
    head_mrr = MRR(relation=relation, direction="s")
    logging.info(f"Pos weight: {pos}, Lr: {lr}, Max epoch: {max_epoch}")

    if args.model == "LinearAggregator":
        nnm = LinearAggregator(relation=relation, sign_constraint=args.sign_constraint)
    elif args.model == "NoisyOrAggregator":
        nnm = NoisyOrAggregator(relation=relation)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    nnm = nnm.to(args.device)
    logging.info(nnm)

    optimizer = torch.optim.Adam(nnm.parameters(), lr=lr)
    train_dataloader = dataloader
    if train_dataloader is None:
        raise ValueError(f"No training data for relation {relation}")

    if args.model == "LinearAggregator":
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos).float())
    elif args.model == "NoisyOrAggregator":
        loss_fn = BCELossR([1, pos])

    for t in range(max_epoch):
        loss = train(train_dataloader, nnm, loss_fn, optimizer, args.noisy_or_reg, unseen)
        nnm.cpu()
        head_mrr.update(nnm, (pos, lr, t))
        tail_mrr.update(nnm, (pos, lr, t))
        nnm.to(args.device)
        max_tail_mrr = tail_mrr.maximums_t_raw
        max_head_mrr = head_mrr.maximums_t_raw
        logging.info(
            f"{relation} tail loss: {loss:>7f} {max_tail_mrr:>7f} {max_head_mrr:>7f} [{t:>5d}/{max_epoch:>5d}]"
        )

    mrr, mrr_raw = calc_mrr(tail_mrr, head_mrr)
    h1, h1_raw = calc_mrr(tail_mrr, head_mrr, "maximums_t_1")
    h10, h10_raw = calc_mrr(tail_mrr, head_mrr, "maximums_t_10")

    metrics = {
        "relation": int(relation),
        "test": {
            "mrr": float(mrr),
            "h1": float(h1),
            "h10": float(h10),
            "mrr_raw": float(mrr_raw),
            "h1_raw": float(h1_raw),
            "h10_raw": float(h10_raw),
        },
    }

    logging.info((mrr, mrr_raw))
    logging.info((h1, h1_raw))
    logging.info((h10, h10_raw))

    save(head_mrr, args.experiment, f"head_mrr_{pos}_{lr}")
    save(tail_mrr, args.experiment, f"tail_mrr_{pos}_{lr}")
    with open(f"{args.experiment}/metric-{relation}.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


if __name__ == "__main__":
    args = get_parser().parse_args()
    args.directory_explanations = f"./{args.dataset}/expl/explanations-processed/"
    args.directory_preprocessed_datasets = f"./{args.dataset}/datasets/"
    time = datetime.now().strftime("%m%d-%H%M")
    args.experiment = f"./{args.dataset}/exp-{time}"

    # Set up experiment folder
    if not os.path.exists(args.experiment):
        os.makedirs(args.experiment)
    # Copy stuff for reproducibility
    shutil.copy(__file__, args.experiment)
    with open(f"{args.experiment}/config.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    # Set up logger
    logging.basicConfig(
        filename=f"{args.experiment}/run.log",
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(ch)
    logging.info(f"Starting experiment {args.experiment}")
    logging.info(pformat(vars(args)))

    c = kge.Config()
    c.set("dataset.name", args.dataset)
    dataset = kge.Dataset.create(c)

    test_sp_to_o = dataset.index("test_sp_to_o")
    test_po_to_s = dataset.index("test_po_to_s")
    test_torch = dataset.split("test")

    valid_sp_to_o = dataset.index("valid_sp_to_o")
    valid_po_to_s = dataset.index("valid_po_to_s")

    processed_sp_test = pickle.load(open(args.directory_explanations + "processed_sp_test.pkl", "rb"))
    processed_po_test = pickle.load(open(args.directory_explanations + "processed_po_test.pkl", "rb"))

    processed_sp_valid = pickle.load(open(args.directory_explanations + "processed_sp_valid.pkl", "rb"))
    processed_po_valid = pickle.load(open(args.directory_explanations + "processed_po_valid.pkl", "rb"))

    rule_map = pickle.load(open(args.directory_explanations + "rule_map.pkl", "rb"))
    rule_features = pickle.load(open(args.directory_explanations + "rule_features.pkl", "rb"))

    LEN_RULES = len(rule_features)
    PAD_TOK = LEN_RULES

    metrics = aggregate_single(args.relation)
    logging.info(pformat(metrics))

    logging.info("Done")
