#!/usr/bin/env python
# coding: utf-8
import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
import copy
import ctypes
import itertools
import json
import math
import os
import pickle
import shutil
import uuid
import warnings
from datetime import datetime
from os.path import exists
from pprint import pformat
from time import perf_counter

import kge
import numpy as np
import torch
from torch import multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")


STEP_TIMINGS = defaultdict(float)
STEP_COUNTS = defaultdict(int)
STEP_GPU_REQUIRED = {
    "load_dataloaders": False,
    "epoch_train.batch_regularization": True,
    "epoch_train.batch_to_device": False,
    "epoch_train.batch_forward_backward": True,
    "epoch_train": True,
    "epoch_model_to_cpu": False,
    "epoch_eval_head": False,
    "epoch_eval_tail": False,
    "epoch_model_to_device": False,
    "epoch_eval.rank_prepare_tensors": False,
    "epoch_eval.rank_model_infer": True,
    "epoch_eval.rank_rankcalc": False,
    "save_outputs": False,
}


@contextmanager
def step_timer(step_name):
    start = perf_counter()
    try:
        yield
    finally:
        STEP_TIMINGS[step_name] += perf_counter() - start
        STEP_COUNTS[step_name] += 1


def print_step_profile():
    if len(STEP_TIMINGS) == 0:
        return

    total = sum([v for k, v in STEP_TIMINGS.items() if '.' not in k])
    print("\n===== Step Timing Summary =====")
    print("step_name,total_seconds,calls,avg_seconds,gpu_required")

    for step_name, seconds in sorted(STEP_TIMINGS.items(), key=lambda x: x[1], reverse=True):
        calls = STEP_COUNTS[step_name]
        avg = seconds / max(calls, 1)
        gpu_required = STEP_GPU_REQUIRED.get(step_name, "unknown")
        print(f"{step_name},{seconds:.6f},{calls},{avg:.6f},{gpu_required}")

    print(f"TOTAL_PROFILED_SECONDS,{total:.6f}")
    print("===== End Step Timing Summary =====\n")


def save(obj, folder, name=None, override=False):
    if name is None:
        name = uuid.uuid4()
    if not os.path.exists(folder):
        os.makedirs(folder)
    path_to_file = f"{folder}/{name}"
    if exists(path_to_file):
        print(f"Warning name {name} exists in cache, do you want to overwrite y/n?")
        confirm = input() if not override else "y"
        if confirm != "y":
            return None

    pickle.dump(obj, open(path_to_file, "wb"))
    return name


def load(folder, name):
    path_to_file = f"{folder}/{name}"
    if exists(path_to_file):
        return pickle.load(open(f"{folder}/{name}", "rb"))
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
                with step_timer("epoch_train.batch_regularization"):
                    rule_confs = torch.nn.functional.sigmoid(model.rules.weight)
                    sudo_false = torch.zeros_like(rule_confs)
                    loss = loss_fn(rule_confs, sudo_false) / dataloader.batch_size
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        # Compute prediction error

        with step_timer("epoch_train.batch_to_device"):
            rules = rules.long().to(args.device)
            y = y.to(args.device)

        with step_timer("epoch_train.batch_forward_backward"):
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
    return test_loss


def rank_batch(nnm, golds, candidates, rules, test_filter):

    batch_rank, batch_rank_raw = [], []
    if len(candidates) > 0 and len(rules) > 0:
        fill_value = 0.0
        scores = torch.full((dataset.num_entities(),), fill_value)
        scores_raw = torch.full((dataset.num_entities(),), fill_value)

        # 这里的 rules 已经在 get_ranks() 中预先构造成 padded tensor 并缓存，
        # 避免在每个 epoch / 每个 key 的 rank_batch 中重复构造。
        rules_ = rules
        if rules_.numel() == 0:
            return torch.tensor(batch_rank), torch.tensor(batch_rank_raw), len(golds)

        with step_timer("epoch_eval.rank_model_infer"):
            with torch.no_grad():
                pred = nnm(rules_).detach()
                if args.model != "NoisyOrAggregator":
                    pred = torch.sigmoid(pred).detach()

        # 优化点：用张量查表替代 np.vectorize(get_conf) + cpu/numpy 往返。
        # RULE_CONF_TABLE[rule_id] 直接给出该 rule 的 confidence，PAD_TOK 对应 0。
        # 这样可把每次 eval 的规则置信度计算下沉到纯 torch 张量运算，减少 Python/NumPy 开销。
        max_conf = RULE_CONF_TABLE[rules_].max(dim=1, keepdim=True).values

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
                # 优化点：只计算当前 gold 的 rank，而不是每次对全实体做 scipy.rankdata 全量排名。
                # 这里用计数法复现 rankdata(method="average") 的排名定义：
                # rank = #(score < gold_score) + ( #(score == gold_score) + 1 ) / 2
                # 复杂度从“每个 gold 做一次全量排序”降到“每个 gold 做一次线性计数比较”，
                # 可显著降低 eval 的 CPU 时间。
                with step_timer("epoch_eval.rank_rankcalc"):
                    gold_score = scores[gold]
                    n_less = (scores < gold_score).sum().item()
                    n_equal = (scores == gold_score).sum().item()
                    rank = n_less + (n_equal + 1) / 2.0
                batch_rank.append(rank)
                scores[gold] = fill_value
            return batch_rank

        batch_rank = get_rank(scores)
        batch_rank_raw = get_rank(scores_raw)

    return torch.tensor(batch_rank), torch.tensor(batch_rank_raw), len(golds)


def build_relation_key_index(index_dict, direction="o"):
    relation_to_keys = defaultdict(list)
    if direction == "o":
        for key in index_dict.keys():
            relation_to_keys[key[1]].append(key)
    else:
        for key in index_dict.keys():
            relation_to_keys[key[0]].append(key)
    return relation_to_keys


def get_ranks(nnm, sp_to_o, processed, relation, direction="o", filter_test=False):
    nnm.eval()
    # 优化点：直接使用全局 relation_keys 索引，避免每次 get_ranks 线性扫描所有 keys。
    split_name = "valid" if filter_test else "test"
    direction_name = "o" if direction == "o" else "s"
    keys = relation_keys[f"{split_name}_{direction_name}"].get(relation, [])

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
        rules = torch.empty((0, 0), dtype=torch.long)
        if key in processed:
            candidates = processed[key]["candidates"]

            # 优化点：对每个 key 的规则列表只做一次 nested->padded 构造并缓存。
            # 原实现会在 rank_batch() 中每次 eval 都重复执行：
            # [torch.tensor(x) for x in rules] + nested_tensor + to_padded_tensor
            # 这是典型 CPU 热点。缓存后后续 epoch 直接复用张量，显著降低 rank_prepare_tensors 时间。
            if "rules_padded_tensor" not in processed[key]:
                with step_timer("epoch_eval.rank_prepare_tensors"):
                    rule_lists = processed[key]["rules"]
                    if len(rule_lists) > 0:
                        processed[key]["rules_padded_tensor"] = torch.nested.to_padded_tensor(
                            torch.nested.nested_tensor([torch.tensor(x) for x in rule_lists]), padding=PAD_TOK
                        ).long()
                    else:
                        processed[key]["rules_padded_tensor"] = torch.empty((0, 0), dtype=torch.long)

            rules = processed[key]["rules_padded_tensor"]
        data.append((nnm, golds, candidates, rules, test_filter))

    data = itertools.starmap(rank_batch, data)
    rank, rank_raw, ns = zip(*data)
    return torch.hstack(rank), torch.hstack(rank_raw), sum(ns)


class LinearAggregator(nn.Module):
    def init_weights(self):
        with torch.no_grad():
            torch.manual_seed(0)
            confs = RULE_CONF_TABLE[torch.tensor(self.relation_rule_ids, dtype=torch.long)].reshape(-1, 1)
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
            confs = RULE_CONF_TABLE[torch.tensor(self.relation_rule_ids, dtype=torch.long)].reshape(-1, 1)
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


def compact_mrr_for_save(mrr_obj):
    mrr_light = copy.copy(mrr_obj)

    # Drop large references to dataset/processed structures
    mrr_light.valid_sp_to_o = None
    mrr_light.valid_processed = None
    mrr_light.test_sp_to_o = None
    mrr_light.test_processed = None

    # Keep only model parameters instead of full model objects
    if mrr_light.nnm is not None:
        mrr_light.nnm = {k: v.detach().cpu() for k, v in mrr_light.nnm.state_dict().items()}
    if mrr_light.nnm_raw is not None:
        mrr_light.nnm_raw = {k: v.detach().cpu() for k, v in mrr_light.nnm_raw.state_dict().items()}

    return mrr_light


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
    with step_timer("load_dataloaders"):
        train_set, _, _ = load(dataset_directory, f"dataset_{relation}.p")

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
        "--model",
        action="store",
        help="Aggregator to use; one of ['LinearAggregator', 'NoisyOrAggregator']",
        default="LinearAggregator",
    )
    parser.add_argument("--shuffle_train", action="store_true", help="Shuffles the examples before creating batches")
    parser.add_argument("--batch_size", action="store", help="Size of batch", default=4096, type=int)
    parser.add_argument("--lr", action="store", default=0.001, help="Learning rates of the adam optimizer", type=float)
    parser.add_argument("--max_epoch", action="store", default=10, help="Epochs to run for each learning rate", type=int)
    parser.add_argument("--pos", action="store", default=15, help="Scaling of the loss for positive examples", type=int)
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
    if args.model == "LinearAggregator":
        nnm = LinearAggregator(relation=relation, sign_constraint=args.sign_constraint)
    elif args.model == "NoisyOrAggregator":
        nnm = NoisyOrAggregator(relation=relation)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    nnm = nnm.to(args.device)

    optimizer = torch.optim.Adam(nnm.parameters(), lr=lr)
    train_dataloader = dataloader
    if train_dataloader is None:
        raise ValueError(f"No training data for relation {relation}")

    if args.model == "LinearAggregator":
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos).float())
    elif args.model == "NoisyOrAggregator":
        loss_fn = BCELossR([1, pos])

    pbar = tqdm(range(max_epoch), desc=f"r{relation}", leave=False)
    for t in pbar:
        with step_timer("epoch_train"):
            loss = train(train_dataloader, nnm, loss_fn, optimizer, args.noisy_or_reg, unseen)
        with step_timer("epoch_model_to_cpu"):
            nnm.cpu()
        with step_timer("epoch_eval_head"):
            head_mrr.update(nnm, (pos, lr, t))
        with step_timer("epoch_eval_tail"):
            tail_mrr.update(nnm, (pos, lr, t))
        with step_timer("epoch_model_to_device"):
            nnm.to(args.device)
        max_tail_mrr = tail_mrr.maximums_t_raw
        max_head_mrr = head_mrr.maximums_t_raw
        max_mrr = (max_tail_mrr + max_head_mrr) / 2
        pbar.set_postfix(tail_loss=f"{loss:.5f}", max_mrr=f"{max_mrr:.5f}")

    mrr, mrr_raw = calc_mrr(tail_mrr, head_mrr)
    h1, h1_raw = calc_mrr(tail_mrr, head_mrr, "maximums_t_1")
    h10, h10_raw = calc_mrr(tail_mrr, head_mrr, "maximums_t_10")

    relation_rule_ids = sorted(rule_map.get(relation, []))
    learned_weights = []
    if len(relation_rule_ids) > 0:
        with torch.no_grad():
            local_weights = nnm.rules.weight[: nnm.num_relation_rules, 0].detach().cpu().numpy()
            if args.model == "LinearAggregator" and args.sign_constraint:
                local_weights = np.square(local_weights)
            elif args.model == "NoisyOrAggregator":
                local_weights = 1 / (1 + np.exp(-local_weights))
            learned_weights = list(zip(relation_rule_ids, local_weights.tolist()))

    num_test_samples = int(test_torch[test_torch[:, 1] == relation].shape[0])
    num_relation_rules = int(len(relation_rule_ids))

    metrics = {
        "relation": int(relation),
        "num_test_samples": num_test_samples,
        "num_relation_rules": num_relation_rules,
        "test": {
            "mrr": float(mrr),
            "h1": float(h1),
            "h10": float(h10),
            "mrr_raw": float(mrr_raw),
            "h1_raw": float(h1_raw),
            "h10_raw": float(h10_raw),
        },
    }

    with step_timer("save_outputs"):
        save((compact_mrr_for_save(head_mrr), compact_mrr_for_save(tail_mrr)), args.experiment, f"mrr-{relation}.pkl")
        with open(f"{args.experiment}/metric-{relation}.json", "w") as f:
            json.dump(metrics, f, indent=4)

        with open(f"{args.experiment}/weight-{relation}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ruleID", "weight"])
            writer.writerows(learned_weights)

    return metrics

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

# 优化点：预构建规则置信度查表，替代 eval 阶段的 np.vectorize(get_conf) 重复计算。
# 约定最后一个位置 PAD_TOK 的置信度为 0。
rule_conf_values = [0.0] * LEN_RULES
for rule_id in range(LEN_RULES):
    # rule_features 可能是 dict（遍历时返回 key:int），因此这里按 id 索引取值，
    # 避免出现“int object is not subscriptable”。
    rule = rule_features[rule_id]
    rule_conf_values[rule_id] = int(rule[1]) / (int(rule[0]) + 5)
RULE_CONF_TABLE = torch.tensor(rule_conf_values + [0.0], dtype=torch.float32)

# 优化点：预构建 relation -> keys 索引，避免每次 get_ranks 线性扫描所有 keys。
relation_keys = {
    "valid_o": build_relation_key_index(valid_sp_to_o, direction="o"),
    "valid_s": build_relation_key_index(valid_po_to_s, direction="s"),
    "test_o": build_relation_key_index(test_sp_to_o, direction="o"),
    "test_s": build_relation_key_index(test_po_to_s, direction="s"),
}

if __name__ == "__main__":
    metrics = aggregate_single(args.relation)
    print(pformat(metrics))
    print_step_profile()
