#!/usr/bin/env python
# coding: utf-8
import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
import copy
import glob
import json
import math
import os
import pickle
import re
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
from tqdm import tqdm

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")


STEP_TIMINGS = defaultdict(float)
STEP_COUNTS = defaultdict(int)
STEP_GPU_REQUIRED = {
    "load_dataloaders": False,
    "epoch_train.iter_create": False,
    "epoch_train.iter_finalize": False,
    "epoch_train.batch_data_wait": False,
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
    "epoch_eval.rank_rankcalc": True,
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


def timed_dataloader_batches(dataloader):
    with step_timer("epoch_train.iter_create"):
        data_iter = iter(dataloader)

    try:
        while True:
            with step_timer("epoch_train.batch_data_wait"):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
            yield batch
    finally:
        with step_timer("epoch_train.iter_finalize"):
            pass


def train(dataloader, model, loss_fn, optimizer, reg=False, num_unseen=0):
    model.train()
    train_loss = 0
    n_loss = 0
    for i, (rules, y) in enumerate(timed_dataloader_batches(dataloader)):

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

        if getattr(dataloader, "on_device", False):
            rules_ = rules
            y_ = y
        else:
            with step_timer("epoch_train.batch_to_device"):
                rules_ = rules.long().to(args.device, non_blocking=True)
                y_ = y.to(args.device, non_blocking=True)

        with step_timer("epoch_train.batch_forward_backward"):
            pred = model(rules_)
            loss = loss_fn(pred.reshape(-1, 1), y_)

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

            if args.model in ["NoisyOrAggregator", "SurprisalAggregator"]:
                pred_prob = pred
            else:
                pred_prob = torch.sigmoid(pred)
            correct += ((pred_prob > 0.5) == y.to(args.device)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return test_loss


def _rank_from_scores_tensor(scores_tensor, golds_t, test_filter_t, fill_value=0.0):
    neg_scores = -1.0 * scores_tensor
    gold_scores = neg_scores[golds_t].clone()

    base_scores = neg_scores.clone()
    base_scores[golds_t] = fill_value
    if test_filter_t is not None:
        base_scores[test_filter_t] = fill_value

    num_golds = int(golds_t.shape[0])
    if num_golds == 0:
        return torch.empty((0,), dtype=torch.float32, device=scores_tensor.device)

    # 对每个 gold 直接做比较计数，避免每个 key 的全排序开销。
    # 这里不做分块：单个 key 下 gold 通常较少，直接一次性计算更简洁。
    pairwise_cmp = base_scores.unsqueeze(0)
    gold_scores_col = gold_scores.unsqueeze(1)
    n_less = (pairwise_cmp < gold_scores_col).sum(dim=1).float()
    n_equal = (pairwise_cmp == gold_scores_col).sum(dim=1).float()

    fill_t = torch.tensor(fill_value, device=scores_tensor.device)
    n_less = n_less - (fill_t < gold_scores).float()
    n_equal = n_equal + 1.0 - (fill_t == gold_scores).float()
    ranks = n_less + (n_equal + 1.0) / 2.0
    return ranks


def rank_batch_group(nnm, batch_items):
    """
    batch_items: list of (golds, candidates, rules, test_filter)
    Returns list of (rank, rank_raw, n)
    """
    model_device = next(nnm.parameters()).device
    if model_device.type == "cpu":
        raise RuntimeError("GPU-only eval is enabled, but model is on CPU")

    fill_value = 0.0
    num_entities = dataset.num_entities()

    outputs = [None] * len(batch_items)

    non_empty_positions = []
    non_empty_rules = []
    non_empty_candidate_lens = []

    for i, (golds_t, candidates_t, rules_t, _test_filter_t) in enumerate(batch_items):
        n = len(golds_t)
        if len(candidates_t) == 0 or len(rules_t) == 0:
            empty = torch.empty((0,), dtype=torch.float32, device=model_device)
            outputs[i] = (empty, empty, n)
            continue

        non_empty_positions.append(i)
        non_empty_rules.append(rules_t)
        non_empty_candidate_lens.append(int(candidates_t.shape[0]))

    if len(non_empty_positions) == 0:
        return outputs

    # Pad rules across keys in this group so we can run one forward pass.
    max_rule_len = max(int(r.shape[1]) for r in non_empty_rules)
    padded_rules = []
    for r in non_empty_rules:
        if int(r.shape[1]) == max_rule_len:
            padded_rules.append(r)
        else:
            pad_cols = max_rule_len - int(r.shape[1])
            pad_block = torch.full((int(r.shape[0]), pad_cols), PAD_TOK, dtype=r.dtype, device=r.device)
            padded_rules.append(torch.cat([r, pad_block], dim=1))
    rules_all = torch.cat(padded_rules, dim=0)

    with step_timer("epoch_eval.rank_model_infer"):
        with torch.no_grad():
            pred_all = nnm(rules_all).detach()
            if args.model not in ["NoisyOrAggregator", "SurprisalAggregator"]:
                pred_all = torch.sigmoid(pred_all).detach()
    max_conf_all = RULE_CONF_TABLE[rules_all].max(dim=1, keepdim=True).values
    score_all = (pred_all * max_conf_all).squeeze(dim=1)
    score_raw_all = pred_all.squeeze(dim=1)

    score_chunks = torch.split(score_all, non_empty_candidate_lens, dim=0)
    score_raw_chunks = torch.split(score_raw_all, non_empty_candidate_lens, dim=0)

    with step_timer("epoch_eval.rank_rankcalc"):
        for chunk_ix, pos in enumerate(non_empty_positions):
            golds_t, candidates_t, _rules_t, test_filter_t = batch_items[pos]
            scores = torch.full((num_entities,), fill_value, device=model_device)
            scores_raw = torch.full((num_entities,), fill_value, device=model_device)

            scores[candidates_t] = score_chunks[chunk_ix]
            scores_raw[candidates_t] = score_raw_chunks[chunk_ix]

            rank = _rank_from_scores_tensor(scores, golds_t, test_filter_t, fill_value=fill_value)
            rank_raw = _rank_from_scores_tensor(scores_raw, golds_t, test_filter_t, fill_value=fill_value)
            outputs[pos] = (rank, rank_raw, len(golds_t))

    return outputs


def build_relation_key_index(index_dict, direction="o"):
    relation_to_keys = defaultdict(list)
    if direction == "o":
        for key in index_dict.keys():
            relation_to_keys[key[1]].append(key)
    else:
        for key in index_dict.keys():
            relation_to_keys[key[0]].append(key)
    return relation_to_keys


def read_ids(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read().splitlines()
    return [line.split("\t")[1] for line in raw]


def split_rule_line(line: str):
    parts = line.rstrip("\n").split("\t")
    if len(parts) >= 4:
        return parts
    return re.split(r"\s+", line.strip(), maxsplit=3)


def extract_head_relation(rule_str: str):
    head = rule_str.split(" <= ", 1)[0].strip()
    if "(" not in head:
        return ""
    return head.split("(", 1)[0].strip()


def parse_rule_file_metadata(rule_file: str, relation_ids):
    relation_to_id = {rel: idx for idx, rel in enumerate(relation_ids)}
    rule_map = defaultdict(list)
    rule_conf_by_id = {}
    rule_relation_by_id = {}
    num_rules = 0
    max_rule_id = 0

    with open(rule_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            parts = split_rule_line(line)
            if len(parts) < 4:
                continue

            num_rules += 1
            max_rule_id = line_no

            try:
                num_preds = int(float(parts[0].strip()))
                num_true = int(float(parts[1].strip()))
            except Exception:
                num_preds = 0
                num_true = 0
            conf = (num_true / (num_preds + 5)) if num_preds >= 0 else 0.0
            rule_conf_by_id[line_no] = float(conf)

            rel = extract_head_relation(parts[3].strip())
            rel_id = relation_to_id.get(rel)
            if rel_id is not None:
                rule_map[rel_id].append(int(line_no))
                rule_relation_by_id[int(line_no)] = int(rel_id)

    return {
        "rule_map": dict(rule_map),
        "rule_conf_by_id": rule_conf_by_id,
        "rule_relation_by_id": rule_relation_by_id,
        "num_rules": int(num_rules),
        "max_rule_id": int(max_rule_id),
    }


def parse_synergy_file(synergy_file: str, rule_relation_by_id):
    synergy_by_relation = defaultdict(list)
    if not synergy_file or (not os.path.exists(synergy_file)):
        return {}

    seen_pairs = defaultdict(set)
    with open(synergy_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                parts = re.split(r"\s+", line)
            if len(parts) < 5:
                continue

            try:
                lift = float(parts[2])
                id1 = int(parts[3])
                id2 = int(parts[4])
            except Exception:
                continue

            if lift <= 0:
                continue

            rel1 = rule_relation_by_id.get(id1)
            rel2 = rule_relation_by_id.get(id2)
            if rel1 is None or rel2 is None or rel1 != rel2:
                continue

            a, b = (id1, id2) if id1 <= id2 else (id2, id1)
            if (a, b) in seen_pairs[rel1]:
                continue
            seen_pairs[rel1].add((a, b))
            synergy_by_relation[rel1].append((a, b, float(lift)))

    return dict(synergy_by_relation)


def get_ranks(nnm, sp_to_o, processed, relation, direction="o", filter_test=False):
    nnm.eval()
    # 优化点：直接使用全局 relation_keys 索引，避免每次 get_ranks 线性扫描所有 keys。
    split_name = "valid" if filter_test else "test"
    direction_name = "o" if direction == "o" else "s"
    keys = relation_keys[f"{split_name}_{direction_name}"].get(relation, [])

    if len(keys) == 0:
        empty = torch.empty((0,), dtype=torch.float32, device=EVAL_DEVICE)
        return empty, empty, 0

    data = []
    for key in keys:
        test_filter = None
        if filter_test:
            if direction == "o":
                if key in test_sp_to_o.keys():
                    test_filter = test_sp_to_o[key].long().to(EVAL_DEVICE, non_blocking=True)
            else:
                if key in test_po_to_s.keys():
                    test_filter = test_po_to_s[key].long().to(EVAL_DEVICE, non_blocking=True)

        golds = sp_to_o[key].long().to(EVAL_DEVICE, non_blocking=True)
        candidates = torch.empty((0,), dtype=torch.long, device=EVAL_DEVICE)
        rules = torch.empty((0, 0), dtype=torch.long, device=EVAL_DEVICE)
        if key in processed:
            if "candidates_tensor_gpu" not in processed[key]:
                processed[key]["candidates_tensor_gpu"] = torch.as_tensor(
                    processed[key]["candidates"], dtype=torch.long, device=EVAL_DEVICE
                )
            candidates = processed[key]["candidates_tensor_gpu"]

            # 优化点：对每个 key 的规则列表只做一次 nested->padded 构造并缓存。
            # 否则每次 eval 都会重复执行：
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
            if "rules_padded_tensor_gpu" not in processed[key]:
                with step_timer("epoch_eval.rank_prepare_tensors"):
                    processed[key]["rules_padded_tensor_gpu"] = processed[key]["rules_padded_tensor"].to(
                        EVAL_DEVICE, non_blocking=True
                    )

            rules = processed[key]["rules_padded_tensor_gpu"]
        data.append((golds, candidates, rules, test_filter))

    results = []
    key_batch_size = max(int(args.eval_key_batch_size), 1)
    for start in range(0, len(data), key_batch_size):
        end = min(start + key_batch_size, len(data))
        group = data[start:end]
        results.extend(rank_batch_group(nnm, group))

    rank, rank_raw, ns = zip(*results)
    return torch.hstack(rank), torch.hstack(rank_raw), sum(ns)


class LinearAggregator(nn.Module):
    def init_weights(self):
        with torch.no_grad():
            torch.manual_seed(0)
            confs = RULE_CONF_TABLE_CPU[torch.tensor(self.relation_rule_ids, dtype=torch.long)].reshape(-1, 1)
            if self.sign_constraint:
                confs = torch.sqrt(torch.clamp(confs, min=0.0))
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

        global_to_local = torch.full((PAD_TOK + 1,), self.pad_local_tok, dtype=torch.long)
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


class SurprisalAggregator(nn.Module):
    def init_weights(self):
        with torch.no_grad():
            torch.manual_seed(0)
            confs = RULE_CONF_TABLE_CPU[torch.tensor(self.relation_rule_ids, dtype=torch.long)].reshape(-1, 1)
            confs = confs.clamp(min=0.0, max=1 - 1e-7)
            surprisal = -torch.log(1 - confs)
            if self.sign_constraint:
                surprisal = torch.sqrt(torch.clamp(surprisal, min=0.0))
            self.rules.weight[: self.num_relation_rules] = surprisal
            if self.num_relation_synergy > 0:
                synergy_init = torch.tensor(self.relation_synergy_lifts, dtype=torch.float32).reshape(-1, 1)
                if self.sign_constraint:
                    synergy_init = torch.sqrt(torch.clamp(synergy_init, min=0.0))
                self.synergy.weight[: self.num_relation_synergy] = synergy_init
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.rules.weight[: self.num_relation_rules].reshape(1, -1))
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            self.bias.uniform_(-bound, bound)
            self.gamma.fill_(1.0)

    def __init__(self, relation, sign_constraint=False):
        super().__init__()
        self.sign_constraint = sign_constraint

        relation_rule_ids = sorted(rule_map.get(relation, []))
        self.relation_rule_ids = np.array(relation_rule_ids, dtype=np.int64)
        self.num_relation_rules = len(relation_rule_ids)
        self.pad_local_tok = self.num_relation_rules

        relation_synergy = sorted(synergy_map.get(relation, []), key=lambda x: (x[0], x[1]))
        local_pairs = []
        local_lifts = []
        global_pairs_filtered = []

        self.rules = nn.Embedding(self.num_relation_rules + 1, 1, padding_idx=self.pad_local_tok)
        self.synergy = nn.Embedding(1, 1, padding_idx=0)
        self.bias = nn.Parameter(torch.zeros(1, 1))
        self.gamma = nn.Parameter(torch.tensor(1.0))

        global_to_local = torch.full((PAD_TOK + 1,), self.pad_local_tok, dtype=torch.long)
        if self.num_relation_rules > 0:
            global_to_local[torch.tensor(relation_rule_ids, dtype=torch.long)] = torch.arange(
                self.num_relation_rules, dtype=torch.long
            )
        self.register_buffer("global_to_local", global_to_local)

        for a, b, lift in relation_synergy:
            local_a = int(global_to_local[a].item())
            local_b = int(global_to_local[b].item())
            if local_a == self.pad_local_tok or local_b == self.pad_local_tok:
                continue
            local_pairs.append((local_a, local_b))
            local_lifts.append(float(lift))
            global_pairs_filtered.append((int(a), int(b)))

        self.num_relation_synergy = len(local_pairs)
        self.pad_synergy_tok = self.num_relation_synergy
        self.synergy = nn.Embedding(self.num_relation_synergy + 1, 1, padding_idx=self.pad_synergy_tok)
        self.relation_synergy_lifts = local_lifts
        self.relation_synergy_pairs_global = global_pairs_filtered

        if self.num_relation_synergy > 0:
            pair_a = torch.tensor([p[0] for p in local_pairs], dtype=torch.long)
            pair_b = torch.tensor([p[1] for p in local_pairs], dtype=torch.long)
        else:
            pair_a = torch.empty((0,), dtype=torch.long)
            pair_b = torch.empty((0,), dtype=torch.long)
        self.register_buffer("synergy_pair_a_local", pair_a)
        self.register_buffer("synergy_pair_b_local", pair_b)

        self.init_weights()

    def forward(self, rules):
        local_rules = self.global_to_local[rules.long()]
        mask = local_rules == self.pad_local_tok
        rule_w = self.rules(local_rules).squeeze(dim=2)
        rule_w.masked_fill_(mask, 0.0)
        if self.sign_constraint:
            rule_w = rule_w**2
        score = rule_w.sum(dim=1, keepdim=True)

        if self.num_relation_synergy > 0:
            batch_size = int(local_rules.shape[0])
            active = ~mask
            active_matrix = torch.zeros(
                (batch_size, self.num_relation_rules), dtype=torch.bool, device=local_rules.device
            )
            row_idx = torch.arange(batch_size, device=local_rules.device).unsqueeze(1).expand_as(local_rules)
            active_matrix[row_idx[active], local_rules[active]] = True

            pair_active = active_matrix[:, self.synergy_pair_a_local] & active_matrix[:, self.synergy_pair_b_local]
            synergy_w = self.synergy.weight[: self.num_relation_synergy, 0].reshape(1, -1)
            if self.sign_constraint:
                synergy_w = synergy_w**2
            synergy_score = (pair_active.float() * synergy_w).sum(dim=1, keepdim=True)
            score = score + self.gamma * synergy_score

        score = score + self.bias
        score = 1 - torch.exp(-score)
        score = torch.clamp(score, min=1e-7, max=1 - 1e-7)
        return score

class NoisyOrAggregator(nn.Module):
    def init_weights(self):
        with torch.no_grad():
            torch.manual_seed(0)
            confs = RULE_CONF_TABLE_CPU[torch.tensor(self.relation_rule_ids, dtype=torch.long)].reshape(-1, 1)
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

        global_to_local = torch.full((PAD_TOK + 1,), self.pad_local_tok, dtype=torch.long)
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

        # Use -1 so the first eval checkpoint is always accepted,
        # even when metric values can be exactly 0.
        self.maximums_v = -1.0
        self.maximums_v_raw = -1.0

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
        if v_mrr > self.maximums_v:
            self.maximums_v = v_mrr
            self.nnm = copy.deepcopy(nnm)
            self.best_hps = hps

        if v_mrr_raw > self.maximums_v_raw:
            self.maximums_v_raw = v_mrr_raw
            self.nnm_raw = copy.deepcopy(nnm)
            self.best_hps_raw = hps

    def finalize_test(self):
        # 只在训练结束后对 best-valid checkpoint 跑一次 test，减少评估调用次数。
        if self.nnm is not None:
            (t_mrr, t_h1, t_h10, _, _, _) = self.calc_metrics(
                self.nnm, self.test_sp_to_o, self.test_processed, direction=self.direction
            )
            self.maximums_t = t_mrr
            self.maximums_t_1 = t_h1
            self.maximums_t_10 = t_h10

        if self.nnm_raw is not None:
            (_, _, _, t_mrr_raw, t_h1_raw, t_h10_raw) = self.calc_metrics(
                self.nnm_raw, self.test_sp_to_o, self.test_processed, direction=self.direction
            )
            self.maximums_t_raw = t_mrr_raw
            self.maximums_t_1_raw = t_h1_raw
            self.maximums_t_10_raw = t_h10_raw


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


class FastTensorBatchLoader:
    def __init__(self, rules, ys, batch_size, shuffle=False, device=None, preload_to_device=False):
        self.rules = rules.contiguous()
        self.ys = ys.contiguous()
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.size = int(ys.shape[0])
        self.on_device = False

        if preload_to_device and device is not None:
            # 一次性把该 relation 的训练数据搬到设备，避免每个 batch 反复 host->device 拷贝。
            self.rules = self.rules.long().to(device, non_blocking=True)
            self.ys = self.ys.to(device, non_blocking=True)
            self.on_device = True

    def __len__(self):
        if self.size == 0:
            return 0
        return (self.size + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.size == 0:
            return
        if self.shuffle:
            perm = torch.randperm(self.size)
            for i in range(0, self.size, self.batch_size):
                idx = perm[i: i + self.batch_size]
                yield self.rules[idx], self.ys[idx]
        else:
            for i in range(0, self.size, self.batch_size):
                yield self.rules[i: i + self.batch_size], self.ys[i: i + self.batch_size]


def materialize_compact_split_to_padded(split_dict):
    offsets = split_dict["offsets"].long()
    rules_flat = split_dict["rules_flat"].int()
    ys = split_dict["golds"].float()

    num_samples = int(ys.shape[0])
    if num_samples == 0:
        return torch.empty((0, 0), dtype=torch.int32), ys

    lengths = offsets[1:] - offsets[:-1]
    max_len = int(lengths.max().item())
    padded = torch.full((num_samples, max_len), PAD_TOK, dtype=torch.int32)

    for i in range(num_samples):
        start = int(offsets[i].item())
        end = int(offsets[i + 1].item())
        n = end - start
        if n > 0:
            padded[i, :n] = rules_flat[start:end]

    return padded, ys


def load_dataloaders(dataset_directory, relation):
    with step_timer("load_dataloaders"):
        data_obj = load(dataset_directory, f"dataset_{relation}.p")

        if not (isinstance(data_obj, dict) and data_obj.get("format") == "compact_varlen_int32_v1"):
            raise ValueError(
                "dataset format is not compact_varlen_int32_v1. "
                "Please regenerate dataset_*.p with updated create_datasets.py"
            )

        train_split = data_obj["train"]
        rules_padded, ys = materialize_compact_split_to_padded(train_split)

        train_loader = FastTensorBatchLoader(
            rules_padded,
            ys,
            batch_size=args.batch_size,
            shuffle=args.shuffle_train,
            device=args.device,
            preload_to_device=args.device != "cpu",
        )

        if len(train_loader) == 0:
            return None
        return train_loader


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", action="store", help="Name of dataset (libkge)", default="codex-m")
    parser.add_argument("-dev", "--device", action="store", help="Device cpu/cuda", default="cuda")
    parser.add_argument(
        "--max_worker_dataloader",
        action="store",
        help="Number of processes for dataloader",
        default=len(os.sched_getaffinity(0)) - 1,
        type=int,
    )
    parser.add_argument(
        "--model",
        action="store",
        help="Aggregator to use; one of ['LinearAggregator', 'SurprisalAggregator', 'NoisyOrAggregator']",
        default="LinearAggregator",
    )
    parser.add_argument("--shuffle_train", action="store_true", help="Shuffles the examples before creating batches")
    parser.add_argument(
        "--batch_size", action="store", help="Size of batch", default=4096, type=int
    )
    parser.add_argument(
        "--lr",
        action="store",
        default="0.01,0.005,0.001",
        help="Learning rate or comma-separated phase learning rates, e.g. 0.01,0.005,0.001",
    )
    parser.add_argument("--max_epoch", action="store", default=60, help="Epochs to run for each learning rate", type=int)
    parser.add_argument(
        "--evaluate_every",
        action="store",
        default="4,2,1",
        help="Evaluation interval or comma-separated phase intervals, e.g. 4,2,1. Use 0 for no eval in a phase.",
    )
    parser.add_argument(
        "--early_stopping",
        action="store",
        default=5,
        type=int,
        help="Stop if valid metric does not improve for X consecutive evaluations. -1 disables.",
    )
    parser.add_argument("--pos", action="store", default=5, help="Scaling of the loss for positive examples", type=int)
    parser.add_argument(
        "--no_sign_constraint",
        dest="sign_constraint",
        action="store_false",
        help="Disable sign constraint. By default, sign constraint is enabled for LinearAggregator.",
    )
    parser.set_defaults(sign_constraint=True)
    parser.add_argument(
        "--noisy_or_reg", action="store_true", help="Sudo negative examples for noisy-or learning.", default=False
    )
    parser.add_argument("--num_unseen", help="Num Sudo negative examples for noisy-or learning.", default=0, type=int)
    parser.add_argument("--relation", action="store", help="Relation to train on", default=0, type=int)
    parser.add_argument(
        "--multiprocess",
        action="store",
        help="Number of processes for all-relation run. 0/1 means single-process.",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--eval_key_batch_size",
        action="store",
        default=64,
        type=int,
        help="How many eval keys to group into one model inference call.",
    )
    parser.add_argument(
        "--rule_file",
        action="store",
        default="",
        help="Path to rules file. Default: ./<dataset>/rules/rules-1000",
    )
    parser.add_argument(
        "--synergy_file",
        action="store",
        default="",
        help="Path to synergy file. Default: ./<dataset>/rules/synergy.txt",
    )

    return parser


def parse_csv_schedule(raw_value, cast_fn, name):
    parts = [p.strip() for p in str(raw_value).split(",") if p.strip() != ""]
    if len(parts) == 0:
        raise ValueError(f"{name} must not be empty")
    try:
        values = [cast_fn(p) for p in parts]
    except Exception as e:
        raise ValueError(f"Invalid {name}: {raw_value}") from e
    return values


def build_phase_lengths(max_epoch, num_phases):
    max_epoch = int(max_epoch)
    num_phases = int(num_phases)
    if max_epoch <= 0:
        return []
    if num_phases <= 0:
        raise ValueError("num_phases must be positive")
    if num_phases > max_epoch:
        raise ValueError(f"num_phases ({num_phases}) cannot exceed max_epoch ({max_epoch})")

    if num_phases == 3:
        return [int(max_epoch * 0.4), int(max_epoch * 0.4), max_epoch - int(max_epoch * 0.8)]

    base = max_epoch // num_phases
    rem = max_epoch % num_phases
    return [base + (1 if i < rem else 0) for i in range(num_phases)]


def phase_value_for_epoch(epoch_idx, phase_lengths, phase_values):
    cursor = 0
    for i, phase_len in enumerate(phase_lengths):
        next_cursor = cursor + phase_len
        if epoch_idx < next_cursor:
            local_epoch = epoch_idx - cursor + 1
            return i, local_epoch, phase_values[i]
        cursor = next_cursor
    i = len(phase_values) - 1
    return i, phase_lengths[-1], phase_values[i]


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
    relation_start_time = perf_counter()
    load_start_time = perf_counter()
    dataloader = load_dataloaders(args.directory_preprocessed_datasets, relation)
    load_seconds = perf_counter() - load_start_time

    pos = args.pos
    lr_values = parse_csv_schedule(args.lr, float, "lr")
    if any(v <= 0 for v in lr_values):
        raise ValueError(f"All lr values must be > 0, got {lr_values}")
    max_epoch = args.max_epoch
    unseen = args.num_unseen
    tail_mrr = MRR(relation=relation, direction="o")
    head_mrr = MRR(relation=relation, direction="s")
    if args.model == "LinearAggregator":
        nnm = LinearAggregator(relation=relation, sign_constraint=args.sign_constraint)
    elif args.model == "SurprisalAggregator":
        nnm = SurprisalAggregator(relation=relation, sign_constraint=args.sign_constraint)
    elif args.model == "NoisyOrAggregator":
        nnm = NoisyOrAggregator(relation=relation)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    nnm = nnm.to(args.device)
    if nnm.rules.weight.device.type == "cpu":
        raise RuntimeError("GPU-only eval requires CUDA device; please set --device cuda")

    relation_rule_ids = sorted(rule_map.get(relation, []))
    with torch.no_grad():
        initial_rule_weights = nnm.rules.weight[: nnm.num_relation_rules, 0].detach().cpu().numpy().copy()
        initial_bias_value = float(nnm.bias.detach().reshape(-1)[0].cpu().item()) if hasattr(nnm, "bias") else None
        initial_gamma_value = float(nnm.gamma.detach().reshape(-1)[0].cpu().item()) if hasattr(nnm, "gamma") else None
    synergy_pairs = []
    initial_synergy_weights = np.array([], dtype=np.float32)
    if args.model == "SurprisalAggregator":
        synergy_pairs = list(getattr(nnm, "relation_synergy_pairs_global", []))
        if getattr(nnm, "num_relation_synergy", 0) > 0:
            with torch.no_grad():
                initial_synergy_weights = (
                    nnm.synergy.weight[: nnm.num_relation_synergy, 0].detach().cpu().numpy().copy()
                )

    optimizer = torch.optim.Adam(nnm.parameters(), lr=lr_values[0])
    train_dataloader = dataloader
    if train_dataloader is None:
        raise ValueError(f"No training data for relation {relation}")

    if args.model == "LinearAggregator":
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos).float())
    elif args.model == "SurprisalAggregator":
        loss_fn = BCELossR([1, pos])
    elif args.model == "NoisyOrAggregator":
        loss_fn = BCELossR([1, pos], apply_sigmoid=True)

    eval_every_values = parse_csv_schedule(args.evaluate_every, int, "evaluate_every")
    if any(v < 0 for v in eval_every_values):
        raise ValueError(f"All evaluate_every values must be >= 0, got {eval_every_values}")

    lr_phase_lengths = build_phase_lengths(max_epoch, len(lr_values))
    eval_phase_lengths = build_phase_lengths(max_epoch, len(eval_every_values))

    evaluate_every = eval_every_values[0]
    early_stopping_patience = int(args.early_stopping)

    best_valid_combined_raw = -1.0
    no_improve_eval_rounds = 0
    epochs_trained = 0
    train_seconds = 0.0
    eval_seconds = 0.0

    pbar = tqdm(range(max_epoch), desc=f"r{relation}", leave=False)
    for t in pbar:
        epochs_trained = t + 1

        lr_phase_idx, _lr_local_epoch, current_lr = phase_value_for_epoch(t, lr_phase_lengths, lr_values)
        eval_phase_idx, eval_local_epoch, current_eval_every = phase_value_for_epoch(
            t, eval_phase_lengths, eval_every_values
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = float(current_lr)

        train_start = perf_counter()
        with step_timer("epoch_train"):
            loss = train(train_dataloader, nnm, loss_fn, optimizer, args.noisy_or_reg, unseen)
        train_seconds += perf_counter() - train_start

        do_eval_in_phase = (current_eval_every > 0) and ((eval_local_epoch % int(current_eval_every)) == 0)
        do_eval = do_eval_in_phase or (t == max_epoch - 1)

        if do_eval:
            eval_start = perf_counter()
            with step_timer("epoch_eval_head"):
                head_mrr.update(nnm, (pos, float(current_lr), t))
            with step_timer("epoch_eval_tail"):
                tail_mrr.update(nnm, (pos, float(current_lr), t))
            eval_seconds += perf_counter() - eval_start

            valid_combined_raw = (head_mrr.maximums_v_raw + tail_mrr.maximums_v_raw) / 2.0
            if valid_combined_raw > best_valid_combined_raw:
                best_valid_combined_raw = valid_combined_raw
                no_improve_eval_rounds = 0
            else:
                no_improve_eval_rounds += 1

            if early_stopping_patience > 0 and no_improve_eval_rounds >= early_stopping_patience:
                pbar.set_postfix(
                    tail_loss=f"{loss:.5f}",
                    max_mrr=f"{((tail_mrr.maximums_v_raw + head_mrr.maximums_v_raw) / 2):.5f}",
                    lr=f"{current_lr:.6g}",
                )
                break

        max_tail_mrr = tail_mrr.maximums_v_raw
        max_head_mrr = head_mrr.maximums_v_raw
        max_mrr = (max_tail_mrr + max_head_mrr) / 2
        pbar.set_postfix(
            # phase=f"{lr_phase_idx + 1}/{len(lr_values)}",
            # eval_every=str(current_eval_every),
            # lr=f"{current_lr:.6g}",
            loss=f"{loss:.5f}",
            max_mrr=f"{max_mrr:.5f}",
        )

    # 训练阶段永远只跑 valid，最后一次性跑 test。
    with step_timer("epoch_eval_head"):
        head_mrr.finalize_test()
    with step_timer("epoch_eval_tail"):
        tail_mrr.finalize_test()

    mrr, mrr_raw = calc_mrr(tail_mrr, head_mrr)
    h1, h1_raw = calc_mrr(tail_mrr, head_mrr, "maximums_t_1")
    h10, h10_raw = calc_mrr(tail_mrr, head_mrr, "maximums_t_10")

    learned_weights = []
    if len(relation_rule_ids) > 0:
        with torch.no_grad():
            trained_rule_weights = nnm.rules.weight[: nnm.num_relation_rules, 0].detach().cpu().numpy()
        learned_weights = list(
            zip(relation_rule_ids, initial_rule_weights.tolist(), trained_rule_weights.tolist())
        )

    learned_synergy_weights = []
    if args.model == "SurprisalAggregator" and len(synergy_pairs) > 0:
        with torch.no_grad():
            trained_synergy_weights = nnm.synergy.weight[: nnm.num_relation_synergy, 0].detach().cpu().numpy()
        learned_synergy_weights = [
            (int(a), int(b), float(o), float(t))
            for (a, b), o, t in zip(synergy_pairs, initial_synergy_weights.tolist(), trained_synergy_weights.tolist())
        ]

    with torch.no_grad():
        trained_bias_value = float(nnm.bias.detach().reshape(-1)[0].cpu().item()) if hasattr(nnm, "bias") else None
        trained_gamma_value = float(nnm.gamma.detach().reshape(-1)[0].cpu().item()) if hasattr(nnm, "gamma") else None

    num_test_samples = int(test_torch[test_torch[:, 1] == relation].shape[0])
    num_relation_rules = int(len(relation_rule_ids))
    num_relation_synergy = int(len(synergy_map.get(relation, [])))

    best_valid_mrr, best_valid_mrr_raw = calc_mrr(tail_mrr, head_mrr, "maximums_v")
    relation_total_seconds = perf_counter() - relation_start_time
    other_seconds = relation_total_seconds - load_seconds - train_seconds - eval_seconds
    if other_seconds < 0:
        other_seconds = 0.0

    metrics = {
        "relation": int(relation),
        "num_test_samples": num_test_samples,
        "num_relation_rules": num_relation_rules,
        "num_relation_synergy": num_relation_synergy,
        "num_relation_features": int(num_relation_rules + num_relation_synergy),
        "train": {
            "max_epoch": int(max_epoch),
            "epochs_trained": int(epochs_trained),
            "evaluate_every": int(evaluate_every),
            "lr_schedule": [float(v) for v in lr_values],
            "lr_phase_epochs": [int(v) for v in lr_phase_lengths],
            "evaluate_every_schedule": [int(v) for v in eval_every_values],
            "evaluate_every_phase_epochs": [int(v) for v in eval_phase_lengths],
            "early_stopping_patience_eval_rounds": int(early_stopping_patience),
        },
        "time_seconds": {
            "total": float(relation_total_seconds),
            "load_dataloaders": float(load_seconds),
            "train": float(train_seconds),
            "eval": float(eval_seconds),
            "other": float(other_seconds),
        },
        "best_valid": {
            "mrr": float(best_valid_mrr),
            "mrr_raw": float(best_valid_mrr_raw),
            "head_mrr": float(head_mrr.maximums_v),
            "tail_mrr": float(tail_mrr.maximums_v),
            "head_mrr_raw": float(head_mrr.maximums_v_raw),
            "tail_mrr_raw": float(tail_mrr.maximums_v_raw),
            "combined_raw": float(best_valid_combined_raw),
        },
        "params": {
            "bias": {
                "original": initial_bias_value,
                "trained": trained_bias_value,
            },
            "gamma": {
                "original": initial_gamma_value,
                "trained": trained_gamma_value,
            },
        },
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
            writer.writerow(["ruleID", "original", "trained"])
            writer.writerows(learned_weights)

        with open(f"{args.experiment}/synergy-{relation}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["rule1ID", "rule2ID", "original", "trained"])
            writer.writerows(learned_synergy_weights)

    return metrics


def _get_all_relations():
    return list(range(dataset.num_relations()))


def _get_relation_test_counts():
    relation_ids = test_torch[:, 1].long().cpu()
    counts = torch.bincount(relation_ids, minlength=dataset.num_relations())
    return {int(i): int(c) for i, c in enumerate(counts.tolist())}


def _merge_metric_files(metric_files, relation_test_counts):
    rows = []
    for path in metric_files:
        with open(path, "r") as f:
            m = json.load(f)
        relation = int(m["relation"])
        test = m["test"]
        rows.append(
            {
                "relation": relation,
                "count": relation_test_counts.get(relation, 0),
                "mrr": float(test["mrr"]),
                "h1": float(test["h1"]),
                "h10": float(test["h10"]),
                "mrr_raw": float(test["mrr_raw"]),
                "h1_raw": float(test["h1_raw"]),
                "h10_raw": float(test["h10_raw"]),
            }
        )

    if not rows:
        return {
            "num_relations": 0,
            "macro": {},
            "weighted_by_test_triples": {},
        }

    keys = ["mrr", "h1", "h10", "mrr_raw", "h1_raw", "h10_raw"]
    weighted_rows = [r for r in rows if r["count"] > 0]
    total_weight = sum(r["count"] for r in weighted_rows)
    if total_weight > 0:
        weighted = {k: float(sum(r[k] * r["count"] for r in weighted_rows) / total_weight) for k in keys}
    else:
        weighted = {k: 0.0 for k in keys}

    return {
        "num_relations": len(rows),
        **weighted,
        "total_test_triples_used_for_weight": int(total_weight),
    }


def _finalize_relation_sweep(failed_relations, relation_test_counts, sweep_seconds):
    metric_files = sorted(glob.glob(os.path.join(args.experiment, "metric-*.json")))
    merged = _merge_metric_files(metric_files, relation_test_counts)

    time_keys = ["total", "load_dataloaders", "train", "eval", "other"]
    summed_time_seconds = {k: 0.0 for k in time_keys}
    for path in metric_files:
        with open(path, "r") as f:
            m = json.load(f)
        metric_time = m.get("time_seconds", {})
        for k in time_keys:
            summed_time_seconds[k] += float(metric_time.get(k, 0.0))

    summed_time_seconds["sweep"] = float(sweep_seconds)

    final_result = {
        "experiment": args.experiment,
        "model": args.model,
        "dataset": args.dataset,
        "failed_relations": failed_relations,
        "summary": merged,
        "time_seconds": summed_time_seconds,
    }

    out_path = os.path.join(args.experiment, "metrics-final.json")
    with open(out_path, "w") as f:
        json.dump(final_result, f, indent=4)

    print(f"Finished relation sweep. failed={len(failed_relations)}")
    print(f"Final summary saved to {out_path}")
    return final_result


def aggregate_all_relations_sequential():
    sweep_start_time = perf_counter()
    relations = _get_all_relations()
    relation_test_counts = _get_relation_test_counts()

    print(f"Start relation sweep (sequential), total relations: {len(relations)}")

    failed_relations = {}

    for relation in relations:
        try:
            aggregate_single(relation)
        except Exception as e:
            failed_relations[int(relation)] = str(e)

    sweep_seconds = perf_counter() - sweep_start_time
    return _finalize_relation_sweep(failed_relations, relation_test_counts, sweep_seconds)


def _run_one_relation(relation):
    # Pool workers are daemonic; they cannot spawn children.
    # So DataLoader must run in-process in each worker.
    args.max_worker_dataloader = 0
    aggregate_single(relation)
    return int(relation)


def aggregate_multiple():
    sweep_start_time = perf_counter()
    # 在多进程 worker 内强制 DataLoader 单进程加载（num_workers=0）
    args.max_worker_dataloader = 0

    relations = _get_all_relations()
    relation_test_counts = _get_relation_test_counts()
    num_processes = min(max(int(args.multiprocess), 2), len(relations)) if relations else 0

    print(f"Start relation sweep, total relations: {len(relations)}, processes: {num_processes}")

    failed_relations = {}

    if relations:
        with mp.get_context("spawn").Pool(processes=num_processes) as pool:
            results = [pool.apply_async(_run_one_relation, (relation,)) for relation in relations]
            for relation, result in zip(relations, results):
                try:
                    result.get()
                except Exception as e:
                    failed_relations[int(relation)] = str(e)

    sweep_seconds = perf_counter() - sweep_start_time
    return _finalize_relation_sweep(failed_relations, relation_test_counts, sweep_seconds)

args = get_parser().parse_args()
EVAL_DEVICE = torch.device(args.device)
args.directory_explanations = f"./{args.dataset}/expl/explanations-processed/"
args.directory_preprocessed_datasets = f"./{args.dataset}/datasets/"
if "EXPERIMENT_DIR" not in os.environ:
    time = datetime.now().strftime("%m%d-%H%M")
    os.environ["EXPERIMENT_DIR"] = f"./{args.dataset}/exp-{time}"
args.experiment = os.environ["EXPERIMENT_DIR"]

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

rule_file = args.rule_file if args.rule_file else f"./{args.dataset}/rules/rules-1000"
synergy_file = args.synergy_file if args.synergy_file else os.path.join(os.path.dirname(rule_file), "synergy.txt")
relation_ids = read_ids(f"./{args.dataset}/relation_ids.del")
rule_meta = parse_rule_file_metadata(rule_file, relation_ids)
synergy_map = parse_synergy_file(synergy_file, rule_meta["rule_relation_by_id"])

LEN_RULES = rule_meta["num_rules"]
MAX_RULE_ID = rule_meta["max_rule_id"]
PAD_TOK = MAX_RULE_ID + 1
rule_map = rule_meta["rule_map"]

# 优化点：预构建规则置信度查表，替代 eval 阶段的 np.vectorize(get_conf) 重复计算。
# 约定最后一个位置 PAD_TOK 的置信度为 0。
rule_conf_values = [0.0] * (PAD_TOK + 1)
for rule_id, conf in rule_meta["rule_conf_by_id"].items():
    rid = int(rule_id)
    if rid < 0 or rid >= PAD_TOK:
        continue
    rule_conf_values[rid] = float(conf)

# PAD_TOK 的置信度保留为 0.0
RULE_CONF_TABLE_CPU = torch.tensor(rule_conf_values, dtype=torch.float32)
if EVAL_DEVICE.type == "cpu":
    raise RuntimeError("This version uses GPU-only eval. Please run with --device cuda")
RULE_CONF_TABLE = RULE_CONF_TABLE_CPU.to(EVAL_DEVICE)

# 优化点：预构建 relation -> keys 索引，避免每次 get_ranks 线性扫描所有 keys。
relation_keys = {
    "valid_o": build_relation_key_index(valid_sp_to_o, direction="o"),
    "valid_s": build_relation_key_index(valid_po_to_s, direction="s"),
    "test_o": build_relation_key_index(test_sp_to_o, direction="o"),
    "test_s": build_relation_key_index(test_po_to_s, direction="s"),
}

if __name__ == "__main__":
    if args.relation == -1:
        if args.multiprocess > 1:
            result = aggregate_multiple()
        else:
            result = aggregate_all_relations_sequential()
        print(json.dumps(result["summary"], indent=2))
    else:
        metrics = aggregate_single(args.relation)
        print(pformat(metrics))
    print_step_profile()
