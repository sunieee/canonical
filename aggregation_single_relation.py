#!/usr/bin/env python
# coding: utf-8
import argparse
import copy
import ctypes
import json
import logging
import math
import os
import time
import resource
from argparse import Namespace
from datetime import datetime
from pprint import pformat

import kge
import numpy as np
import scipy
import torch
from torch import multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader, Dataset


torch.multiprocessing.set_sharing_strategy("file_system")


# ---------- globals initialized in main ----------
args = None
dataset = None
rule_features = None
rule_map = None
PAD_TOK = None
LEN_RULES = None
processed_sp_valid = None
processed_po_valid = None
processed_sp_test = None
processed_po_test = None
valid_sp_to_o = None
valid_po_to_s = None
test_sp_to_o = None
test_po_to_s = None
test_torch = None


def _log_stage(msg):
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    logging.info(f"[STAGE][rss={rss_mb:.1f}MB] {msg}")


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
        self.len = xs.shape[0]

    def __getitem__(self, index):
        return self.shared_x[index], self.shared_y[index]

    def __len__(self):
        return self.len


class LinearAggregator(nn.Module):
    def __init__(self, sign_constraint=False):
        super().__init__()
        self.rules = nn.Embedding(LEN_RULES + 1, 1, padding_idx=PAD_TOK)
        self.bias = nn.Parameter(torch.zeros(dataset.num_relations(), 1))
        self.sign_constraint = sign_constraint
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            torch.manual_seed(0)
            for r in rule_map:
                rules = rule_map[r]
                if len(rules) == 0:
                    continue
                self.rules.weight[rules] = torch.from_numpy(get_conf(np.array(rules))).reshape(-1, 1).float()
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.rules.weight[rules].reshape(1, -1))
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                self.bias[r] = self.bias[r].uniform_(-bound, bound)

    def forward(self, rules, relation):
        mask = rules == PAD_TOK
        rules = self.rules(rules)
        rules.masked_fill_(mask.unsqueeze(dim=2), 0.0)
        if self.sign_constraint:
            rules = rules ** 2
        return rules.sum(dim=1) + self.bias[relation]


class NoisyOrAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.rules = nn.Embedding(LEN_RULES + 1, 1, padding_idx=PAD_TOK)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            torch.manual_seed(0)
            for r in rule_map:
                rules = rule_map[r]
                if len(rules) == 0:
                    continue
                confs = torch.from_numpy(get_conf(np.array(rules))).reshape(-1, 1).float().clamp(1e-6, 1 - 1e-6)
                logit_values = torch.log(confs / (1 - confs))
                self.rules.weight[rules] = logit_values.float()

    def forward(self, rules, relation):
        mask = rules == PAD_TOK
        rules = self.rules(rules)
        rules.masked_fill_(mask.unsqueeze(dim=2), float("-inf"))
        no = 1 - (1 - torch.sigmoid(rules)).prod(dim=1)
        return no.clamp(min=1e-4, max=0.99999)


def BCELossR(weights=(1, 1), reduction="mean", apply_sigmoid=False):
    def loss(input_, target):
        x = input_
        if apply_sigmoid:
            x = torch.sigmoid(x).clamp(min=1e-7, max=1 - 1e-7)
        bce = -weights[1] * target * torch.log(x) - (1 - target) * weights[0] * torch.log(1 - x)
        if reduction == "sum":
            return torch.sum(bce)
        if reduction == "mean":
            return torch.mean(bce)
        return bce

    return loss


def get_conf(x):
    if np.isscalar(x):
        if x == PAD_TOK:
            return 0.0
        rule = rule_features[int(x)]
        return int(rule[1]) / (int(rule[0]) + 5)

    x = np.asarray(x)
    out = np.zeros_like(x, dtype=float)
    mask = x != PAD_TOK
    if mask.any():
        flat = x[mask].astype(int)
        vals = [int(rule_features[idx][1]) / (int(rule_features[idx][0]) + 5) for idx in flat]
        out[mask] = np.array(vals, dtype=float)
    return out


def train_epoch(dataloader, model, loss_fn, optimizer, relation):
    model.train()
    total, n = 0.0, 0
    t0 = time.time()
    for batch_idx, (rules, y) in enumerate(dataloader):
        if batch_idx == 0:
            _log_stage(f"train first batch fetched: rules_shape={tuple(rules.shape)}, y_shape={tuple(y.shape)}")
        rules = rules.to(args.device)
        y = y.to(args.device)
        pred = model(rules, relation)
        loss = loss_fn(pred.reshape(-1, 1), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1

        if batch_idx == 0:
            _log_stage(f"train first batch done, loss={loss.item():.6f}")

        if getattr(args, "debug_batches", -1) > 0 and n >= args.debug_batches:
            _log_stage(f"debug_batches reached: {args.debug_batches}, early stop this epoch")
            break

    _log_stage(f"train_epoch finished, batches={n}, cost={time.time() - t0:.2f}s")
    return total / max(n, 1)


def rank_batch(nnm, golds, candidates, rules, test_filter, relation):
    if len(candidates) == 0 or len(rules) == 0:
        return torch.tensor([]), torch.tensor([]), len(golds)

    fill_value = 0.0
    scores = torch.full((dataset.num_entities(),), fill_value)
    scores_raw = torch.full((dataset.num_entities(),), fill_value)

    rules_ = torch.nested.to_padded_tensor(
        torch.nested.nested_tensor([torch.tensor(x) for x in rules]), padding=PAD_TOK
    ).long()
    if rules_.numel() == 0:
        return torch.tensor([]), torch.tensor([]), len(golds)

    with torch.no_grad():
        pred = nnm(rules_, relation).detach()
        if args.model != "NoisyOrAggregator":
            pred = torch.sigmoid(pred).detach()

    max_conf = get_conf(rules_.cpu().numpy()).max(axis=1).reshape(-1, 1).astype(np.float32)
    scores[candidates] = (pred * torch.tensor(max_conf)).squeeze(dim=1)
    scores_raw[candidates] = pred.squeeze(dim=1)

    def get_rank(local_scores):
        out = []
        local_scores = -1 * local_scores
        gold_scores = local_scores[golds].clone()
        local_scores[golds] = fill_value
        if test_filter is not None:
            local_scores[test_filter] = fill_value
        for ix, gold in enumerate(golds):
            g = gold.item()
            local_scores[g] = gold_scores[ix]
            ranking = scipy.stats.rankdata(local_scores.detach().numpy())
            out.append(ranking[g])
            local_scores[g] = fill_value
        return out

    return torch.tensor(get_rank(scores)), torch.tensor(get_rank(scores_raw)), len(golds)


def get_ranks(nnm, sp_to_o, processed, relation, direction="o", filter_test=False):
    nnm.eval()
    if direction == "o":
        keys = [key for key in sp_to_o.keys() if key[1] == relation]
    else:
        keys = [key for key in sp_to_o.keys() if key[0] == relation]

    if len(keys) == 0:
        return torch.tensor([]), torch.tensor([]), 0

    rank, rank_raw, n_total = [], [], 0
    for key in keys:
        test_filter = None
        if filter_test:
            if direction == "o" and key in test_sp_to_o:
                test_filter = test_sp_to_o[key].long()
            if direction == "s" and key in test_po_to_s:
                test_filter = test_po_to_s[key].long()

        golds = sp_to_o[key].long()
        candidates = processed.get(key, {}).get("candidates", [])
        rules = processed.get(key, {}).get("rules", [])

        r, rr, n = rank_batch(nnm, golds, candidates, rules, test_filter, relation)
        rank.append(r)
        rank_raw.append(rr)
        n_total += n

    if len(rank) == 0:
        return torch.tensor([]), torch.tensor([]), 0
    return torch.hstack(rank), torch.hstack(rank_raw), n_total


def calc_metrics(ranks, n):
    if n == 0 or len(ranks) == 0:
        return 0.0, 0.0, 0.0
    mrr = ((1 / ranks).sum() / n).item()
    h1 = ((ranks == 1.0).sum() / n).item()
    h10 = ((ranks <= 10.0).sum() / n).item()
    return mrr, h1, h10


def evaluate_relation(nnm, relation):
    # tail
    v_tail, v_tail_raw, n_tv = get_ranks(nnm, valid_sp_to_o, processed_sp_valid, relation, "o", filter_test=True)
    t_tail, t_tail_raw, n_tt = get_ranks(nnm, test_sp_to_o, processed_sp_test, relation, "o", filter_test=False)
    # head
    v_head, v_head_raw, n_hv = get_ranks(nnm, valid_po_to_s, processed_po_valid, relation, "s", filter_test=True)
    t_head, t_head_raw, n_ht = get_ranks(nnm, test_po_to_s, processed_po_test, relation, "s", filter_test=False)

    metrics = {
        "valid_tail": dict(zip(["mrr", "hits@1", "hits@10"], calc_metrics(v_tail, n_tv))),
        "valid_head": dict(zip(["mrr", "hits@1", "hits@10"], calc_metrics(v_head, n_hv))),
        "valid_tail_raw": dict(zip(["mrr", "hits@1", "hits@10"], calc_metrics(v_tail_raw, n_tv))),
        "valid_head_raw": dict(zip(["mrr", "hits@1", "hits@10"], calc_metrics(v_head_raw, n_hv))),
        "test_tail": dict(zip(["mrr", "hits@1", "hits@10"], calc_metrics(t_tail, n_tt))),
        "test_head": dict(zip(["mrr", "hits@1", "hits@10"], calc_metrics(t_head, n_ht))),
        "test_tail_raw": dict(zip(["mrr", "hits@1", "hits@10"], calc_metrics(t_tail_raw, n_tt))),
        "test_head_raw": dict(zip(["mrr", "hits@1", "hits@10"], calc_metrics(t_head_raw, n_ht))),
    }

    metrics["valid_combined_mrr"] = (
        metrics["valid_tail"]["mrr"] + metrics["valid_head"]["mrr"]
    ) / 2
    metrics["valid_combined_mrr_raw"] = (
        metrics["valid_tail_raw"]["mrr"] + metrics["valid_head_raw"]["mrr"]
    ) / 2
    metrics["test_combined_mrr"] = (
        metrics["test_tail"]["mrr"] + metrics["test_head"]["mrr"]
    ) / 2
    metrics["test_combined_mrr_raw"] = (
        metrics["test_tail_raw"]["mrr"] + metrics["test_head_raw"]["mrr"]
    ) / 2

    return metrics


def load_pickle(path):
    import pickle

    t0 = time.time()
    _log_stage(f"pickle loading: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    _log_stage(f"pickle loaded: {path}, cost={time.time()-t0:.2f}s")
    return obj


def remap_rule_tensor(rule_tensor, old_to_new, old_pad, new_pad):
    rule_tensor = rule_tensor.clone()
    pad_mask = rule_tensor == old_pad
    if (~pad_mask).any():
        vals = rule_tensor[~pad_mask].cpu().numpy().astype(int)
        mapped = np.array([old_to_new[int(v)] for v in vals], dtype=np.int64)
        rule_tensor[~pad_mask] = torch.from_numpy(mapped).to(rule_tensor.device)
    rule_tensor[pad_mask] = new_pad
    return rule_tensor


def remap_processed_rules(processed, old_to_new):
    out = {}
    for k, v in processed.items():
        new_v = dict(v)
        if "rules" in new_v:
            new_v["rules"] = [[old_to_new[int(r)] for r in row] for row in new_v["rules"]]
        out[k] = new_v
    return out


def load_relation_resources(relation):
    global rule_features, rule_map, PAD_TOK, LEN_RULES
    global processed_sp_valid, processed_po_valid, processed_sp_test, processed_po_test

    t0 = time.time()
    _log_stage(f"load dataset_{relation}.p")
    dataset_tuple = load_pickle(f"{args.directory_preprocessed_datasets}/dataset_{relation}.p")
    train_set, valid_set, test_set = dataset_tuple
    _log_stage(f"dataset_{relation}.p loaded in {time.time() - t0:.2f}s")

    t1 = time.time()
    _log_stage("load rule_map/rule_features")
    full_rule_map = load_pickle(args.directory_explanations + "rule_map.pkl")
    full_rule_features = load_pickle(args.directory_explanations + "rule_features.pkl")
    _log_stage(f"rule metadata loaded in {time.time() - t1:.2f}s")

    t2 = time.time()
    _log_stage("load processed_sp_valid.pkl")
    full_processed_sp_valid = load_pickle(args.directory_explanations + "processed_sp_valid.pkl")
    _log_stage(f"processed_sp_valid loaded in {time.time() - t2:.2f}s")

    t2 = time.time()
    _log_stage("load processed_po_valid.pkl")
    full_processed_po_valid = load_pickle(args.directory_explanations + "processed_po_valid.pkl")
    _log_stage(f"processed_po_valid loaded in {time.time() - t2:.2f}s")

    t2 = time.time()
    _log_stage("load processed_sp_test.pkl")
    full_processed_sp_test = load_pickle(args.directory_explanations + "processed_sp_test.pkl")
    _log_stage(f"processed_sp_test loaded in {time.time() - t2:.2f}s")

    t2 = time.time()
    _log_stage("load processed_po_test.pkl")
    full_processed_po_test = load_pickle(args.directory_explanations + "processed_po_test.pkl")
    _log_stage(f"processed_po_test loaded in {time.time() - t2:.2f}s")

    old_pad = len(full_rule_features)

    used_rule_ids = set(full_rule_map.get(relation, []))

    def collect_from_tensor_dataset(ds):
        rid_t = torch.vstack((ds.datasets[0].tensors[3], ds.datasets[1].tensors[3]))
        used_rule_ids.update([int(x) for x in rid_t[rid_t != old_pad].unique().tolist()])

    collect_from_tensor_dataset(train_set)
    collect_from_tensor_dataset(valid_set)
    collect_from_tensor_dataset(test_set)

    for name, processed_dict in (
        ("sp_valid", full_processed_sp_valid),
        ("po_valid", full_processed_po_valid),
        ("sp_test", full_processed_sp_test),
        ("po_test", full_processed_po_test),
    ):
        t_scan = time.time()
        hit_keys = 0
        for key, val in processed_dict.items():
            if (len(key) > 1 and key[1] == relation) or (len(key) > 0 and key[0] == relation):
                hit_keys += 1
                for row in val.get("rules", []):
                    used_rule_ids.update([int(r) for r in row])
        _log_stage(f"scan processed {name} done, hit_keys={hit_keys}, cost={time.time()-t_scan:.2f}s")

    _log_stage(f"collect used_rule_ids done, count={len(used_rule_ids)}")
    used_rule_ids = sorted(used_rule_ids)
    old_to_new = {old: i for i, old in enumerate(used_rule_ids)}

    rule_features = [full_rule_features[old] for old in used_rule_ids]
    LEN_RULES = len(rule_features)
    PAD_TOK = LEN_RULES
    _log_stage(f"build rule remap done, LEN_RULES={LEN_RULES}, PAD_TOK={PAD_TOK}")

    rule_map = {relation: [old_to_new[int(r)] for r in full_rule_map.get(relation, []) if int(r) in old_to_new]}

    def remap_ds(ds):
        x = torch.vstack((ds.datasets[0].tensors[3], ds.datasets[1].tensors[3]))
        y = torch.vstack((ds.datasets[0].tensors[4], ds.datasets[1].tensors[4]))
        x = remap_rule_tensor(x, old_to_new, old_pad, PAD_TOK)
        return SharedDataset(x, y)

    _log_stage("start remap train/valid/test tensors to relation-local rule ids")
    t3 = time.time()
    train_shared = remap_ds(train_set)
    _log_stage(f"train remap done, n={len(train_shared)}, cost={time.time()-t3:.2f}s")
    t3 = time.time()
    valid_shared = remap_ds(valid_set)
    _log_stage(f"valid remap done, n={len(valid_shared)}, cost={time.time()-t3:.2f}s")
    t3 = time.time()
    test_shared = remap_ds(test_set)
    _log_stage(f"test remap done, n={len(test_shared)}, cost={time.time()-t3:.2f}s")

    t4 = time.time()
    _log_stage("start remap processed_sp/po valid/test by relation")
    processed_sp_valid = remap_processed_rules(
        {k: v for k, v in full_processed_sp_valid.items() if k[1] == relation}, old_to_new
    )
    processed_po_valid = remap_processed_rules(
        {k: v for k, v in full_processed_po_valid.items() if k[0] == relation}, old_to_new
    )
    processed_sp_test = remap_processed_rules(
        {k: v for k, v in full_processed_sp_test.items() if k[1] == relation}, old_to_new
    )
    processed_po_test = remap_processed_rules(
        {k: v for k, v in full_processed_po_test.items() if k[0] == relation}, old_to_new
    )

    _log_stage(f"processed remap done, cost={time.time()-t4:.2f}s")
    _log_stage("build DataLoader(train/valid/test)")
    train_loader = DataLoader(
        train_shared, batch_size=args.batch_size, shuffle=args.shuffle_train, num_workers=args.max_worker_dataloader
    )
    valid_loader = DataLoader(valid_shared, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_shared, batch_size=args.batch_size, shuffle=False)
    _log_stage("DataLoader built")
    return train_loader, valid_loader, test_loader


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config-base.json")
    parser.add_argument("-d", "--dataset", default="codex-m")
    parser.add_argument("-dev", "--device", default="cuda")
    parser.add_argument("--relation", type=int, required=True)
    parser.add_argument("--max_worker_dataloader", type=int, default=0)
    parser.add_argument("--model", default="LinearAggregator", choices=["LinearAggregator", "NoisyOrAggregator"])
    parser.add_argument("--shuffle_train", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_epoch", type=int, default=20)
    parser.add_argument("--pos", type=float, default=15)
    parser.add_argument("--sign_constraint", action="store_true")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--debug_batches", type=int, default=0)
    parser.add_argument("--force_dataloader_workers_0", action="store_true", default=True)
    return parser


def run_single_relation_experiment(run_args=None):
    global args, dataset
    global valid_sp_to_o, valid_po_to_s, test_sp_to_o, test_po_to_s, test_torch

    if run_args is None:
        run_args = get_parser().parse_args()

    args = copy.deepcopy(run_args)
    if args.config is not None and os.path.exists(args.config):
        with open(args.config) as f:
            config = json.load(f)
        args_dict = vars(args)
        # assert set(config.keys()).issubset(args_dict.keys()), "config 文件存在未识别参数"
        args = Namespace(**{**args_dict, **config})

    args.directory_explanations = f"./{args.dataset}/expl/explanations-processed/"
    args.directory_preprocessed_datasets = f"./{args.dataset}/datasets/"

    # 强制单进程 DataLoader，避免在高 RSS 下 worker fork 导致 OOM(137)
    if getattr(args, "force_dataloader_workers_0", False):
        args.max_worker_dataloader = 0

    if args.output_dir is None:
        time_str = datetime.now().strftime("%m%d-%H%M%S")
        args.output_dir = f"./{args.dataset}/exp-relation-{args.relation}-{time_str}"
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    logging.info("Run single relation experiment")
    logging.info(pformat(vars(args)))

    c = kge.Config()
    c.set("dataset.name", args.dataset)
    _log_stage("before kge.Dataset.create")
    dataset = kge.Dataset.create(c)
    _log_stage("after kge.Dataset.create")

    _log_stage("build dataset indexes test_sp_to_o/test_po_to_s/test split")
    test_sp_to_o = dataset.index("test_sp_to_o")
    test_po_to_s = dataset.index("test_po_to_s")
    test_torch = dataset.split("test")
    _log_stage("test indexes built")

    _log_stage("build dataset indexes valid_sp_to_o/valid_po_to_s")
    valid_sp_to_o = dataset.index("valid_sp_to_o")
    valid_po_to_s = dataset.index("valid_po_to_s")
    _log_stage("valid indexes built")

    _log_stage(f"start load_relation_resources relation={args.relation}")
    train_loader, _, _ = load_relation_resources(args.relation)
    _log_stage(f"load_relation_resources done relation={args.relation}")

    if len(train_loader.dataset) == 0:
        result = {
            "relation": args.relation,
            "config": vars(args),
            "status": "skipped_empty_dataset",
        }
        out_file = os.path.join(args.output_dir, f"relation_{args.relation}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    _log_stage("start build model/loss")
    if args.model == "LinearAggregator":
        model = LinearAggregator(sign_constraint=args.sign_constraint)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos).float())
    else:
        model = NoisyOrAggregator()
        loss_fn = BCELossR([1, args.pos])
    _log_stage("model/loss built")

    _log_stage(f"move model to device={args.device}")
    model = model.to(args.device)
    _log_stage("model moved to device")

    _log_stage("build optimizer")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    _log_stage("optimizer built")

    best = {
        "valid_score": -1.0,
        "epoch": -1,
        "metrics": None,
        "model_state": None,
    }

    for epoch in range(args.max_epoch):
        loss = train_epoch(train_loader, model, loss_fn, optimizer, args.relation)

        model_cpu = copy.deepcopy(model).cpu()
        metrics = evaluate_relation(model_cpu, args.relation)
        valid_score = max(metrics["valid_combined_mrr"], metrics["valid_combined_mrr_raw"])

        if valid_score > best["valid_score"]:
            best["valid_score"] = valid_score
            best["epoch"] = epoch
            best["metrics"] = metrics
            best["model_state"] = copy.deepcopy(model_cpu.state_dict())

        logging.info(
            f"relation={args.relation} epoch={epoch + 1}/{args.max_epoch} loss={loss:.6f} "
            f"valid_mrr={metrics['valid_combined_mrr']:.6f} valid_mrr_raw={metrics['valid_combined_mrr_raw']:.6f}"
        )

    out_file = os.path.join(args.output_dir, f"relation_{args.relation}.json")
    result = {
        "relation": args.relation,
        "config": vars(args),
        "best_epoch": best["epoch"],
        "best_valid_score": best["valid_score"],
        "metrics": best["metrics"],
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if best["model_state"] is not None:
        torch.save(best["model_state"], os.path.join(args.output_dir, f"relation_{args.relation}.pt"))

    return result


if __name__ == "__main__":
    run_single_relation_experiment()
