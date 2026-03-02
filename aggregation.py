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
from argparse import Namespace
from collections import defaultdict
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
from tqdm import tqdm

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


def train(dataloader, model, loss_fn, optimizer, scheduler, relation, reg=False, num_unseen=0):
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

        rules = rules.to(args.device)
        y = y.to(args.device)
        pred = model(rules, relation)
        loss = loss_fn(pred.reshape(-1, 1), y)

        train_loss += loss.item()
        n_loss += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss / n_loss


def test(dataloader, model, loss_fn, relation):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for i, (rules, y) in enumerate(dataloader):

            rules = rules.to(args.device)
            y = y.to(args.device)

            pred = model(rules, relation).reshape(-1, 1)

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


def rank_batch(nnm, golds, candidates, rules, test_filter, relation):

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
            pred = nnm(rules_, relation).detach()
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


def get_ranks(nnm, sp_to_o, processed, relation, direction="o", filter_test=False, file=None, fill_value=0.0):
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
        data.append((nnm, golds, candidates, rules, test_filter, relation))

    data = itertools.starmap(rank_batch, data)
    rank, rank_raw, ns = zip(*data)
    return torch.hstack(rank), torch.hstack(rank_raw), sum(ns)


class LinearAggregator(nn.Module):
    def init_weights(self):
        with torch.no_grad():
            for r in rule_map:
                torch.manual_seed(0)
                rules = rule_map[r]
                self.rules.weight[rules] = torch.from_numpy(get_conf(rules)).reshape(-1, 1).float()
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.rules.weight[rules].reshape(1, -1))
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                self.bias[r] = self.bias[r].uniform_(-bound, bound)

    def __init__(self, sign_constraint=False):
        super().__init__()
        self.rules = nn.Embedding(LEN_RULES + 1, 1, padding_idx=PAD_TOK)
        self.bias = nn.Parameter(torch.zeros(dataset.num_relations(), 1))  # len relations, 100
        self.init_weights()
        self.sign_constraint = sign_constraint

    def forward(self, rules, relation):
        mask = rules == PAD_TOK
        rules = self.rules(rules)
        rules.masked_fill_(mask.unsqueeze(dim=2), 0.0)
        if self.sign_constraint:
            rules = rules**2
        logits = rules.sum(dim=1) + self.bias[relation]
        return logits


class NoisyOrAggregator(nn.Module):
    def init_weights(self):
        with torch.no_grad():
            for r in rule_map:
                torch.manual_seed(0)
                rules = rule_map[r]
                confs = torch.from_numpy(get_conf(rules)).reshape(-1, 1).float()
                logit_values = torch.log(confs / (1 - confs))
                self.rules.weight[rules] = logit_values.float()

    def __init__(self):
        super().__init__()
        self.rules = nn.Embedding(LEN_RULES + 1, 1, padding_idx=PAD_TOK)
        self.init_weights()

    def forward(self, rules, relation):
        mask = rules == PAD_TOK
        rules = self.rules(rules)
        rules.masked_fill_(mask.unsqueeze(dim=2), float("-inf"))
        no = 1 - (1 - torch.nn.functional.sigmoid(rules)).prod(dim=1)
        no = no.clamp(min=0.0001, max=0.99999)
        return no


def calc_mrr(tail_mrr, head_mrr, attr="maximums_t"):
    head_rank = 0
    tail_rank = 0
    head_rank_raw = 0
    tail_rank_raw = 0
    n = 0
    for ix in range(dataset.num_relations()):
        rn = test_torch[test_torch[:, 1] == ix].shape[0]
        tail_rank += getattr(tail_mrr, attr).get(ix, 0.0) * rn
        head_rank += getattr(head_mrr, attr).get(ix, 0.0) * rn
        tail_rank_raw += getattr(tail_mrr, attr + "_raw").get(ix, 0.0) * rn
        head_rank_raw += getattr(head_mrr, attr + "_raw").get(ix, 0.0) * rn
        n += rn
    return (head_rank + tail_rank) / (2 * n), (head_rank_raw + tail_rank_raw) / (2 * n)


class MRR:
    def __init__(self, direction="o"):
        self.direction = direction

        self.best_hps = {}
        self.best_hps_raw = {}

        self.maximums_v = defaultdict(float)
        self.maximums_v_raw = defaultdict(float)

        self.maximums_t = defaultdict(float)
        self.maximums_t_raw = defaultdict(float)
        self.maximums_t_1 = defaultdict(float)
        self.maximums_t_1_raw = defaultdict(float)
        self.maximums_t_10 = defaultdict(float)
        self.maximums_t_10_raw = defaultdict(float)

        self.valid_sp_to_o = valid_sp_to_o if direction == "o" else valid_po_to_s
        self.valid_processed = processed_sp_valid if direction == "o" else processed_po_valid
        self.test_sp_to_o = test_sp_to_o if direction == "o" else test_po_to_s
        self.test_processed = processed_sp_test if direction == "o" else processed_po_test
        self.nnm = dict()
        self.nnm_raw = dict()

    def calc_metrics_(self, ranks, n):
        if n == 0:
            return 0.0, 0.0, 0.0
        mrr = ((1 / ranks).sum() / n).item()
        h1 = ((ranks == 1.0).sum() / n).item()
        h10 = ((ranks <= 10.0).sum() / n).item()
        return mrr, h1, h10

    def calc_metrics(self, nnm, sp_to_o, processed, relation, direction, filter_test=False):
        ranks, ranks_raw, n = get_ranks(nnm, sp_to_o, processed, relation, direction, filter_test)
        mrr, h1, h10 = self.calc_metrics_(ranks, n)
        mrr_raw, h1_raw, h10_raw = self.calc_metrics_(ranks_raw, n)
        return (mrr, h1, h10, mrr_raw, h1_raw, h10_raw)

    def update(self, nnm, relation, hps):
        (v_mrr, v_h1, v_h10, v_mrr_raw, v_h1_raw, v_h10_raw) = self.calc_metrics(
            nnm, self.valid_sp_to_o, self.valid_processed, relation, direction=self.direction, filter_test=True
        )
        if (v_mrr > self.maximums_v[relation]) or (v_mrr_raw > self.maximums_v_raw[relation]):
            (t_mrr, t_h1, t_h10, t_mrr_raw, t_h1_raw, t_h10_raw) = self.calc_metrics(
                nnm, self.test_sp_to_o, self.test_processed, relation, direction=self.direction
            )
            if v_mrr > self.maximums_v[relation]:
                self.maximums_v[relation] = v_mrr
                self.maximums_t[relation] = t_mrr
                self.maximums_t_1[relation] = t_h1
                self.maximums_t_10[relation] = t_h10
                self.nnm[relation] = copy.deepcopy(nnm)
                self.best_hps[relation] = hps

            if v_mrr_raw > self.maximums_v_raw[relation]:
                self.maximums_v_raw[relation] = v_mrr_raw
                self.maximums_t_raw[relation] = t_mrr_raw
                self.maximums_t_1_raw[relation] = t_h1_raw
                self.maximums_t_10_raw[relation] = t_h10_raw
                self.nnm_raw[relation] = copy.deepcopy(nnm)
                self.best_hps_raw[relation] = hps


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

    (train_set, valid_set, test_set) = load(dataset_directory, f"dataset_{relation}")

    weight_t = (valid_set.datasets[0].tensors[4] == 0).sum() / (valid_set.datasets[0].tensors[4] == 1).sum()
    weight_h = (valid_set.datasets[1].tensors[4] == 0).sum() / (valid_set.datasets[1].tensors[4] == 1).sum()

    train_set = SharedDataset(
        torch.vstack((train_set.datasets[0].tensors[3], train_set.datasets[1].tensors[3])),
        torch.vstack((train_set.datasets[0].tensors[4], train_set.datasets[1].tensors[4])),
    )
    valid_set = SharedDataset(
        torch.vstack((valid_set.datasets[0].tensors[3], valid_set.datasets[1].tensors[3])),
        torch.vstack((valid_set.datasets[0].tensors[4], valid_set.datasets[1].tensors[4])),
    )
    test_set = SharedDataset(
        torch.vstack((test_set.datasets[0].tensors[3], test_set.datasets[1].tensors[3])),
        torch.vstack((test_set.datasets[0].tensors[4], test_set.datasets[1].tensors[4])),
    )

    if len(train_set) == 0:
        return None, None
    train_dataloader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=args.shuffle_train, num_workers=args.max_worker_dataloader
    )
    valid_dataloader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    return (weight_t, weight_h), (train_dataloader, valid_dataloader, test_dataloader)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", action="store", help="Path to config file; ORDER: default->command line->config file",
        default="config-base.json"
    )
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
        help="Aggregator to use; one of ['LinearAggregator', 'PNAAggregator']",
        default="LinearAggregator",
    )
    parser.add_argument("--shuffle_train", action="store_true", help="Shuffles the examples before creating batches")
    parser.add_argument("--batch_size", action="store", help="Size of batch", default=4096)
    parser.add_argument(
        "--lr_hpo", action="store", nargs="+", default=[0.001, 0.01], help="Learning rates of the adam optimizer"
    )
    parser.add_argument(
        "--max_epoch_hpo",
        action="store",
        nargs="+",
        default=[20, 10],
        help="Epochs to run for each learning rate; max_epoch[i] are trained using lr[i]",
    )
    parser.add_argument(
        "--pos_hpo",
        action="store",
        nargs="+",
        default=[5, 15, 30, 100, 400],
        help="Scaling of the loss for positive examples, controls precision/recall",
    )
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


if __name__ == "__main__":
    args = get_parser().parse_args()
    args.directory_explanations = f"./{args.dataset}/expl/explanations-processed/"
    args.directory_preprocessed_datasets = f"./{args.dataset}/datasets/"
    time = datetime.now().strftime("%m%d-%H%M")
    args.experiment = f"./{args.dataset}/exp-{time}"
    if args.config is not None:
        with open(args.config) as f:
            config = json.load(f)
            args_dict = vars(args)
            assert set(config.keys()).issubset(args_dict.keys()), "There are keys in you config file not recognized"
            args = Namespace(**{**args_dict, **config})

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
    valid_torch = dataset.split("valid")

    dataloaders = dict()
    weights = dict()
    for relation in range(dataset.num_relations()):
        weight, dataloader = load_dataloaders(args.directory_preprocessed_datasets, relation)
        dataloaders[relation] = dataloader
        weights[relation] = weight

    processed_sp_test = pickle.load(open(args.directory_explanations + "processed_sp_test.pkl", "rb"))
    processed_po_test = pickle.load(open(args.directory_explanations + "processed_po_test.pkl", "rb"))

    processed_sp_valid = pickle.load(open(args.directory_explanations + "processed_sp_valid.pkl", "rb"))
    processed_po_valid = pickle.load(open(args.directory_explanations + "processed_po_valid.pkl", "rb"))

    rule_map = pickle.load(open(args.directory_explanations + "rule_map.pkl", "rb"))
    rule_features = pickle.load(open(args.directory_explanations + "rule_features.pkl", "rb"))
    ruleid2relid = {ruleid: relid for relid in rule_map for ruleid in rule_map[relid]}

    filter_test = set([tuple(x.tolist()) for x in test_torch])
    filter_valid = set([tuple(x.tolist()) for x in valid_torch])

    LEN_RULES = len(rule_features)
    PAD_TOK = LEN_RULES

    for pos in args.pos_hpo:
        # for unseen in args.num_unseen:
        unseen = args.num_unseen
        for ix, (lr, max_epoch) in enumerate(zip(args.lr_hpo, args.max_epoch_hpo)):
            tail_mrr = MRR(direction="o")
            head_mrr = MRR(direction="s")
            logging.info(f"Pos weight: {pos}, Lr: {lr}, Max epoch: {max_epoch}")
            if args.model == "LinearAggregator":
                nnm = LinearAggregator(sign_constraint=args.sign_constraint)
            elif args.model == "NoisyOrAggregator":
                nnm = NoisyOrAggregator()
            nnm = nnm.to(args.device)
            logging.info(nnm)

            optimizer = torch.optim.Adam(nnm.parameters(), lr=lr)

            for t in range(max_epoch):
                for relation in tqdm(range(dataset.num_relations())):
                    dataloader = dataloaders[relation]
                    weight = weights[relation]
                    if dataloader is None:
                        continue
                    (train_dataloader, valid_dataloader, test_dataloader) = dataloader
                    if train_dataloader is None:
                        continue

                    if args.model == "LinearAggregator":
                        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos).float())
                    elif args.model == "NoisyOrAggregator":
                        loss_fn = BCELossR([1, pos])

                    loss = train(
                        train_dataloader, nnm, loss_fn, optimizer, None, relation, args.noisy_or_reg, unseen
                    )
                    nnm.cpu()
                    head_mrr.update(nnm, relation, (pos, lr, t))
                    tail_mrr.update(nnm, relation, (pos, lr, t))
                    nnm.to(args.device)
                    max_tail_mrr = tail_mrr.maximums_t_raw[relation]
                    max_head_mrr = head_mrr.maximums_t_raw[relation]
                    logging.info(
                        f"{relation} tail loss: {loss:>7f} {max_tail_mrr} {max_head_mrr} [{t:>5d}/{max_epoch:>5d}]"
                    )

            logging.info(calc_mrr(tail_mrr, head_mrr))
            logging.info(calc_mrr(tail_mrr, head_mrr, "maximums_t_1"))
            logging.info(calc_mrr(tail_mrr, head_mrr, "maximums_t_10"))
            save(head_mrr, args.experiment, f"head_mrr_{pos}_{lr}")
            save(tail_mrr, args.experiment, f"tail_mrr_{pos}_{lr}")

    logging.info("Done")
