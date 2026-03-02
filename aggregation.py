#!/usr/bin/env python
# coding: utf-8
import argparse
import json
import logging
import math
import os
import pickle
from argparse import Namespace
from datetime import datetime
from pprint import pformat

import kge
import numpy as np
import scipy
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_parser():
    parser = argparse.ArgumentParser(description="Single relation trainer/evaluator for rule aggregator")
    parser.add_argument("-c", "--config", default=None, help="Optional config JSON")
    parser.add_argument("-d", "--dataset", default="codex-m", help="kge dataset name")
    parser.add_argument("-dev", "--device", default="cuda", help="cpu/cuda")
    parser.add_argument("--relation", type=int, required=True, help="Relation id to train/evaluate")
    parser.add_argument("--model", default="LinearAggregator", choices=["LinearAggregator", "NoisyOrAggregator"])
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--shuffle_train", action="store_true")
    parser.add_argument("--max_worker_dataloader", type=int, default=max(os.cpu_count() - 1, 0))
    parser.add_argument("--lr", type=float,  help="Single learning rate", default=0.01)
    parser.add_argument("--max_epoch", type=int, help="Single epoch count", default=10)
    parser.add_argument("--pos", type=float,  help="Single positive weight", default=5)
    parser.add_argument("--sign_constraint", action="store_true")
    parser.add_argument("--noisy_or_reg", action="store_true", default=False)
    parser.add_argument("--num_unseen", type=int, default=0)
    parser.add_argument("--experiment_root", default=None, help="Base output directory")
    parser.add_argument("--output_json", default=None, help="Output relation json path")
    return parser


class TensorPairDataset(Dataset):
    def __init__(self, x, y):
        self.x = x.long()
        self.y = y.float()

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.x.shape[0]


def make_dataset(split_set):
    x = torch.vstack((split_set.datasets[0].tensors[3], split_set.datasets[1].tensors[3]))
    y = torch.vstack((split_set.datasets[0].tensors[4], split_set.datasets[1].tensors[4]))
    return TensorPairDataset(x, y)


def build_dataloaders(dataset_dir, relation, batch_size, shuffle_train, workers):
    path = os.path.join(dataset_dir, f"dataset_{relation}.p")
    if not os.path.exists(path):
        return None
    train_set, valid_set, test_set = load_pickle(path)
    train_ds = make_dataset(train_set)
    valid_ds = make_dataset(valid_set)
    test_ds = make_dataset(test_set)
    if len(train_ds) == 0:
        return None
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=workers)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_dl, valid_dl, test_dl


class LinearAggregator(nn.Module):
    def __init__(self, num_rules, pad_tok, init_confs, sign_constraint=False):
        super().__init__()
        self.rules = nn.Embedding(num_rules + 1, 1, padding_idx=pad_tok)
        self.bias = nn.Parameter(torch.zeros(1, 1))
        self.sign_constraint = sign_constraint
        with torch.no_grad():
            self.rules.weight[:num_rules] = torch.from_numpy(init_confs).reshape(-1, 1).float()
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.rules.weight[:num_rules].reshape(1, -1))
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            self.bias.uniform_(-bound, bound)

    def forward(self, rules):
        mask = rules == self.rules.padding_idx
        x = self.rules(rules)
        x.masked_fill_(mask.unsqueeze(dim=2), 0.0)
        if self.sign_constraint:
            x = x**2
        return x.sum(dim=1) + self.bias


class NoisyOrAggregator(nn.Module):
    def __init__(self, num_rules, pad_tok, init_confs):
        super().__init__()
        self.rules = nn.Embedding(num_rules + 1, 1, padding_idx=pad_tok)
        with torch.no_grad():
            confs = torch.from_numpy(np.clip(init_confs, 1e-6, 1 - 1e-6)).reshape(-1, 1).float()
            self.rules.weight[:num_rules] = torch.log(confs / (1 - confs))

    def forward(self, rules):
        mask = rules == self.rules.padding_idx
        x = self.rules(rules)
        x.masked_fill_(mask.unsqueeze(dim=2), float("-inf"))
        no = 1 - (1 - torch.sigmoid(x)).prod(dim=1)
        return no.clamp(min=1e-4, max=1 - 1e-5)


def bce_loss_r(weights):
    def loss(input, target):
        input = torch.clamp(input, min=1e-7, max=1 - 1e-7)
        return torch.mean(-weights[1] * target * torch.log(input) - (1 - target) * weights[0] * torch.log(1 - input))

    return loss


def train_one_epoch(dataloader, model, loss_fn, optimizer, device, reg=False, num_unseen=0):
    model.train()
    total = 0.0
    n = 0
    for i, (rules, y) in enumerate(dataloader):
        if reg and num_unseen > 0:
            num_batches = max(len(dataloader), 1)
            unseen = min(num_unseen, num_batches)
            step = max(int(num_batches / unseen), 1)
            if i % step == 0:
                rule_confs = torch.sigmoid(model.rules.weight)
                pseudo_false = torch.zeros_like(rule_confs)
                loss_reg = torch.nn.functional.binary_cross_entropy(rule_confs, pseudo_false)
                optimizer.zero_grad()
                loss_reg.backward()
                optimizer.step()

        rules = rules.to(device)
        y = y.to(device)
        pred = model(rules)
        loss = loss_fn(pred.reshape(-1, 1), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def remap_rule_sequence(seq, global_to_local, pad_tok):
    return [global_to_local.get(int(x), pad_tok) for x in seq]


def rank_batch(model, golds, candidates, rules, test_filter, num_entities, pad_tok, model_name):
    batch_rank = []
    batch_rank_raw = []
    if len(candidates) == 0 or len(rules) == 0:
        return torch.tensor(batch_rank), torch.tensor(batch_rank_raw), len(golds)

    fill_value = 0.0
    scores = torch.full((num_entities,), fill_value)
    scores_raw = torch.full((num_entities,), fill_value)

    rules_t = torch.nested.to_padded_tensor(torch.nested.nested_tensor([torch.tensor(x) for x in rules]), padding=pad_tok).long()
    if rules_t.numel() == 0:
        return torch.tensor(batch_rank), torch.tensor(batch_rank_raw), len(golds)

    with torch.no_grad():
        pred = model(rules_t).detach().cpu()
        if model_name != "NoisyOrAggregator":
            pred = torch.sigmoid(pred)

    max_conf = (rules_t != pad_tok).float().amax(dim=1).reshape(-1, 1)
    scores[candidates] = (pred * max_conf).squeeze(dim=1)
    scores_raw[candidates] = pred.squeeze(dim=1)

    def calc_rank(vals):
        vals = -1 * vals
        out = []
        gold_scores = vals[golds].clone()
        vals[golds] = fill_value
        if test_filter is not None:
            vals[test_filter] = fill_value
        for i, gold in enumerate(golds):
            g = gold.item()
            vals[g] = gold_scores[i]
            ranking = scipy.stats.rankdata(vals.numpy())
            out.append(ranking[g])
            vals[g] = fill_value
        return out

    batch_rank = calc_rank(scores)
    batch_rank_raw = calc_rank(scores_raw)
    return torch.tensor(batch_rank), torch.tensor(batch_rank_raw), len(golds)


def get_ranks(model, sp_to_o, processed, relation, direction, filter_test, test_sp_to_o, test_po_to_s, num_entities, pad_tok, model_name):
    if direction == "o":
        keys = [k for k in sp_to_o.keys() if k[1] == relation]
    else:
        keys = [k for k in sp_to_o.keys() if k[0] == relation]
    if len(keys) == 0:
        return torch.tensor([]), torch.tensor([]), 0

    all_rank = []
    all_rank_raw = []
    n_total = 0

    for key in keys:
        test_filter = None
        if filter_test:
            if direction == "o" and key in test_sp_to_o:
                test_filter = test_sp_to_o[key].long()
            if direction == "s" and key in test_po_to_s:
                test_filter = test_po_to_s[key].long()

        golds = sp_to_o[key].long()
        candidates = []
        rules = []
        if key in processed:
            candidates = processed[key]["candidates"]
            rules = processed[key]["rules"]

        rank, rank_raw, n = rank_batch(model, golds, candidates, rules, test_filter, num_entities, pad_tok, model_name)
        all_rank.append(rank)
        all_rank_raw.append(rank_raw)
        n_total += n

    return torch.hstack(all_rank), torch.hstack(all_rank_raw), n_total


def calc_metrics(ranks, n):
    if n == 0 or len(ranks) == 0:
        return {"mrr": 0.0, "h1": 0.0, "h10": 0.0}
    return {
        "mrr": ((1 / ranks).sum() / n).item(),
        "h1": ((ranks == 1.0).sum() / n).item(),
        "h10": ((ranks <= 10.0).sum() / n).item(),
    }


def main():
    args = get_parser().parse_args()
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        args_dict = vars(args)
        assert set(cfg.keys()).issubset(args_dict.keys())
        args = Namespace(**{**args_dict, **cfg})

    exp_root = args.experiment_root or f"./{args.dataset}/single-rel-{datetime.now().strftime('%m%d-%H%M%S')}"
    os.makedirs(exp_root, exist_ok=True)
    rel_dir = os.path.join(exp_root, f"relation_{args.relation}")
    os.makedirs(rel_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(rel_dir, "run.log"),
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(pformat(vars(args)))

    c = kge.Config()
    c.set("dataset.name", args.dataset)
    dataset = kge.Dataset.create(c)

    rel = args.relation
    test_sp_to_o = dataset.index("test_sp_to_o")
    test_po_to_s = dataset.index("test_po_to_s")
    valid_sp_to_o = dataset.index("valid_sp_to_o")
    valid_po_to_s = dataset.index("valid_po_to_s")

    expl_dir = f"./{args.dataset}/expl/explanations-processed"
    ds_dir = f"./{args.dataset}/datasets"

    dataloaders = build_dataloaders(ds_dir, rel, args.batch_size, args.shuffle_train, args.max_worker_dataloader)
    if dataloaders is None:
        out_path = args.output_json or os.path.join(exp_root, f"relation_{rel}.json")
        with open(out_path, "w") as f:
            json.dump({"relation": rel, "skipped": True, "reason": "empty_or_missing_dataset"}, f, indent=2)
        return

    train_dl, _, _ = dataloaders

    processed_sp_test = load_pickle(os.path.join(expl_dir, "processed_sp_test.pkl"))
    processed_po_test = load_pickle(os.path.join(expl_dir, "processed_po_test.pkl"))
    processed_sp_valid = load_pickle(os.path.join(expl_dir, "processed_sp_valid.pkl"))
    processed_po_valid = load_pickle(os.path.join(expl_dir, "processed_po_valid.pkl"))

    rule_map = load_pickle(os.path.join(expl_dir, "rule_map.pkl"))
    rule_features = load_pickle(os.path.join(expl_dir, "rule_features.pkl"))

    relation_rules_global = rule_map.get(rel, [])
    global_to_local = {rid: i for i, rid in enumerate(relation_rules_global)}
    local_rule_count = len(relation_rules_global)
    pad_tok = local_rule_count

    if local_rule_count == 0:
        out_path = args.output_json or os.path.join(exp_root, f"relation_{rel}.json")
        with open(out_path, "w") as f:
            json.dump({"relation": rel, "skipped": True, "reason": "no_rules_for_relation"}, f, indent=2)
        return

    confs = []
    for rid in relation_rules_global:
        rule = rule_features[rid]
        confs.append(int(rule[1]) / (int(rule[0]) + 5))
    confs = np.array(confs, dtype=np.float32)

    # 将训练 batch 中的全局 rule id 映射到单 relation 局部 rule id
    def remap_batch_rules(batch_rules):
        remapped = batch_rules.clone()
        for i in range(batch_rules.shape[0]):
            for j in range(batch_rules.shape[1]):
                remapped[i, j] = global_to_local.get(int(batch_rules[i, j].item()), pad_tok)
        return remapped

    class RemapLoader:
        def __init__(self, inner_loader):
            self.inner_loader = inner_loader
            self.batch_size = inner_loader.batch_size

        def __iter__(self):
            for rules, y in self.inner_loader:
                yield remap_batch_rules(rules), y

        def __len__(self):
            return len(self.inner_loader)

    train_dl = RemapLoader(train_dl)

    def remap_processed(proc):
        out = {}
        for k, v in proc.items():
            if (k[1] == rel) or (k[0] == rel):
                rr = [remap_rule_sequence(seq, global_to_local, pad_tok) for seq in v["rules"]]
                out[k] = {"candidates": v["candidates"], "rules": rr}
        return out

    processed_sp_valid_rel = remap_processed(processed_sp_valid)
    processed_po_valid_rel = remap_processed(processed_po_valid)
    processed_sp_test_rel = remap_processed(processed_sp_test)
    processed_po_test_rel = remap_processed(processed_po_test)

    if args.model == "LinearAggregator":
        model = LinearAggregator(local_rule_count, pad_tok, confs, args.sign_constraint)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos).float())
    else:
        model = NoisyOrAggregator(local_rule_count, pad_tok, confs)
        loss_fn = bce_loss_r([1, args.pos])

    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = {
        "valid_mrr": -1.0,
        "valid_mrr_raw": -1.0,
        "epoch": -1,
        "metrics": None,
    }

    for epoch in range(args.max_epoch):
        loss = train_one_epoch(train_dl, model, loss_fn, optimizer, args.device, args.noisy_or_reg, args.num_unseen)

        model_cpu = model.cpu()
        v_tail, v_tail_raw, n_vt = get_ranks(
            model_cpu, valid_sp_to_o, processed_sp_valid_rel, rel, "o", True,
            test_sp_to_o, test_po_to_s, dataset.num_entities(), pad_tok, args.model
        )
        v_head, v_head_raw, n_vh = get_ranks(
            model_cpu, valid_po_to_s, processed_po_valid_rel, rel, "s", True,
            test_sp_to_o, test_po_to_s, dataset.num_entities(), pad_tok, args.model
        )
        t_tail, t_tail_raw, n_tt = get_ranks(
            model_cpu, test_sp_to_o, processed_sp_test_rel, rel, "o", False,
            test_sp_to_o, test_po_to_s, dataset.num_entities(), pad_tok, args.model
        )
        t_head, t_head_raw, n_th = get_ranks(
            model_cpu, test_po_to_s, processed_po_test_rel, rel, "s", False,
            test_sp_to_o, test_po_to_s, dataset.num_entities(), pad_tok, args.model
        )

        valid_tail = calc_metrics(v_tail, n_vt)
        valid_head = calc_metrics(v_head, n_vh)
        valid_tail_raw = calc_metrics(v_tail_raw, n_vt)
        valid_head_raw = calc_metrics(v_head_raw, n_vh)

        test_tail = calc_metrics(t_tail, n_tt)
        test_head = calc_metrics(t_head, n_th)
        test_tail_raw = calc_metrics(t_tail_raw, n_tt)
        test_head_raw = calc_metrics(t_head_raw, n_th)

        valid_mrr = (valid_head["mrr"] + valid_tail["mrr"]) / 2
        valid_mrr_raw = (valid_head_raw["mrr"] + valid_tail_raw["mrr"]) / 2

        current = {
            "epoch": epoch,
            "train_loss": loss,
            "valid": {
                "head": valid_head,
                "tail": valid_tail,
                "mean": {
                    "mrr": valid_mrr,
                    "h1": (valid_head["h1"] + valid_tail["h1"]) / 2,
                    "h10": (valid_head["h10"] + valid_tail["h10"]) / 2,
                },
                "head_raw": valid_head_raw,
                "tail_raw": valid_tail_raw,
                "mean_raw": {
                    "mrr": valid_mrr_raw,
                    "h1": (valid_head_raw["h1"] + valid_tail_raw["h1"]) / 2,
                    "h10": (valid_head_raw["h10"] + valid_tail_raw["h10"]) / 2,
                },
            },
            "test": {
                "head": test_head,
                "tail": test_tail,
                "mean": {
                    "mrr": (test_head["mrr"] + test_tail["mrr"]) / 2,
                    "h1": (test_head["h1"] + test_tail["h1"]) / 2,
                    "h10": (test_head["h10"] + test_tail["h10"]) / 2,
                },
                "head_raw": test_head_raw,
                "tail_raw": test_tail_raw,
                "mean_raw": {
                    "mrr": (test_head_raw["mrr"] + test_tail_raw["mrr"]) / 2,
                    "h1": (test_head_raw["h1"] + test_tail_raw["h1"]) / 2,
                    "h10": (test_head_raw["h10"] + test_tail_raw["h10"]) / 2,
                },
            },
            "counts": {
                "valid_head": n_vh,
                "valid_tail": n_vt,
                "test_head": n_th,
                "test_tail": n_tt,
            },
        }

        if (valid_mrr > best["valid_mrr"]) or (valid_mrr_raw > best["valid_mrr_raw"]):
            best = {
                "valid_mrr": max(best["valid_mrr"], valid_mrr),
                "valid_mrr_raw": max(best["valid_mrr_raw"], valid_mrr_raw),
                "epoch": epoch,
                "metrics": current,
            }

        model = model_cpu.to(args.device)
        logging.info(f"relation={rel} epoch={epoch} loss={loss:.6f} valid_mrr={valid_mrr:.6f} valid_mrr_raw={valid_mrr_raw:.6f}")

    output = {
        "relation": rel,
        "config": {
            "dataset": args.dataset,
            "model": args.model,
            "lr": args.lr,
            "max_epoch": args.max_epoch,
            "pos": args.pos,
            "device": args.device,
            "sign_constraint": args.sign_constraint,
            "noisy_or_reg": args.noisy_or_reg,
            "num_unseen": args.num_unseen,
        },
        "best_epoch": best["epoch"],
        "result": best["metrics"],
        "skipped": False,
    }

    out_path = args.output_json or os.path.join(exp_root, f"relation_{rel}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logging.info(f"Saved result to {out_path}")


if __name__ == "__main__":
    main()
