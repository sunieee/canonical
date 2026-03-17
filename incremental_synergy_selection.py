#!/usr/bin/env python
# coding: utf-8

import argparse
import csv
import json
import os
import pickle
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

import kge


# Compatibility placeholder for legacy pickles created from aggregation.py
# where MRR was serialized under __main__.MRR.
class MRR:
    pass


class _LegacyMRRUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "MRR":
            return MRR
        return super().find_class(module, name)


def read_ids(file_path: str) -> List[str]:
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


def parse_rule_file_metadata(rule_file: str, relation_ids: List[str]):
    relation_to_id = {rel: idx for idx, rel in enumerate(relation_ids)}
    rule_map = {}
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
                rule_map.setdefault(rel_id, []).append(int(line_no))
                rule_relation_by_id[int(line_no)] = int(rel_id)

    return {
        "rule_map": rule_map,
        "rule_conf_by_id": rule_conf_by_id,
        "rule_relation_by_id": rule_relation_by_id,
        "num_rules": int(num_rules),
        "max_rule_id": int(max_rule_id),
    }


def parse_synergy_file_for_relation(
    synergy_file: str,
    rule_relation_by_id: Dict[int, int],
    relation: int,
    min_synergy: float,
    min_supp: int,
) -> List[Tuple[int, int, float, int]]:
    """
    Returns list of (a, b, lift, support) for one relation.
    """
    if not synergy_file or (not os.path.exists(synergy_file)):
        return []

    seen = set()
    out = []
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
                supp = int(parts[1])
                lift = float(parts[2])
                id1 = int(parts[3])
                id2 = int(parts[4])
            except Exception:
                continue

            if supp < int(min_supp):
                continue
            if lift < float(min_synergy):
                continue

            rel1 = rule_relation_by_id.get(id1)
            rel2 = rule_relation_by_id.get(id2)
            if rel1 is None or rel2 is None or rel1 != rel2 or rel1 != relation:
                continue

            a, b = (id1, id2) if id1 <= id2 else (id2, id1)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            out.append((a, b, float(lift), int(supp)))

    out.sort(key=lambda x: (x[0], x[1]))
    return out


def save_pickle(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except AttributeError:
            f.seek(0)
            return _LegacyMRRUnpickler(f).load()


def build_compact_split(sp_to_o, processed_sp, relation, direction="o"):
    rules_flat = []
    offsets = [0]
    golds = []

    for key in sp_to_o.keys():
        if direction == "o":
            _e, r = key
        else:
            r, _e = key

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

    return {
        "rules_flat": torch.tensor(rules_flat, dtype=torch.int32),
        "offsets": torch.tensor(offsets, dtype=torch.int64),
        "golds": torch.tensor(golds, dtype=torch.float32).reshape(-1, 1),
        "num_samples": int(len(golds)),
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


@dataclass
class BaseParams:
    relation_rule_ids: List[int]
    local_weights: torch.Tensor
    bias: float


def _load_state_dict_from_checkpoint(path: str):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise ValueError(f"Unsupported checkpoint format: {path}")


def load_base_params(args, relation_rule_ids: List[int]) -> BaseParams:
    num_relation_rules = len(relation_rule_ids)

    if args.mrr_pickle:
        mrr_head, mrr_tail = load_pickle(args.mrr_pickle)
        state = None
        # Prefer raw-best checkpoint first.
        if getattr(mrr_tail, "nnm_raw", None) is not None:
            state = mrr_tail.nnm_raw
        elif getattr(mrr_tail, "nnm", None) is not None:
            state = mrr_tail.nnm
        elif getattr(mrr_head, "nnm_raw", None) is not None:
            state = mrr_head.nnm_raw
        elif getattr(mrr_head, "nnm", None) is not None:
            state = mrr_head.nnm
        if state is None:
            raise ValueError("No model state found inside mrr pickle")
    elif args.checkpoint:
        state = _load_state_dict_from_checkpoint(args.checkpoint)
    else:
        raise ValueError("Please provide --mrr_pickle or --checkpoint")

    if "rules.weight" not in state:
        raise ValueError("Checkpoint missing key 'rules.weight'")

    rule_w = state["rules.weight"]
    if isinstance(rule_w, torch.nn.Parameter):
        rule_w = rule_w.data
    rule_w = torch.as_tensor(rule_w, dtype=torch.float32)

    if rule_w.shape[0] < num_relation_rules:
        raise ValueError(
            f"Checkpoint rules.weight rows ({rule_w.shape[0]}) < relation rules ({num_relation_rules})"
        )

    local_w = rule_w[:num_relation_rules, 0].clone().float()

    bias = 0.0
    if "bias" in state:
        b = torch.as_tensor(state["bias"], dtype=torch.float32).reshape(-1)
        if b.numel() > 0:
            bias = float(b[0].item())

    return BaseParams(relation_rule_ids=relation_rule_ids, local_weights=local_w, bias=bias)


def precompute_base_logits_and_active_rules(
    split_dict,
    base: BaseParams,
    rule_global_to_local: Dict[int, int],
    clamp_w_min=0.0,
    clamp_w_max=7.0,
):
    offsets = split_dict["offsets"].long()
    rules_flat = split_dict["rules_flat"].long()
    y = split_dict["golds"].float().reshape(-1)

    n = int(y.shape[0])
    z0 = torch.zeros((n,), dtype=torch.float32)
    active_rule_sets = [set() for _ in range(n)]

    for i in range(n):
        start = int(offsets[i].item())
        end = int(offsets[i + 1].item())
        if end <= start:
            z0[i] = float(base.bias)
            continue

        local_ids = []
        for rid in rules_flat[start:end].tolist():
            rid_i = int(rid)
            local = rule_global_to_local.get(rid_i, None)
            if local is None:
                continue
            local_ids.append(local)
            active_rule_sets[i].add(rid_i)

        if len(local_ids) == 0:
            score = 0.0
        else:
            lw = base.local_weights[torch.tensor(local_ids, dtype=torch.long)]
            lw = torch.clamp(lw, min=clamp_w_min, max=clamp_w_max)
            score = float(lw.sum().item())

        z0[i] = float(score + base.bias)

    return {
        "z0": z0,
        "y": y,
        "active_rule_sets": active_rule_sets,
    }


def predict_prob_from_logits(logits: torch.Tensor) -> torch.Tensor:
    surprisal = F.softplus(logits)
    p = 1.0 - torch.exp(-torch.clamp(surprisal, min=0.0))
    p = torch.clamp(p, min=1e-7, max=1 - 1e-7)
    return p


def bce_loss_from_logits(logits: torch.Tensor, y: torch.Tensor, pos_weight: float = 1.0) -> torch.Tensor:
    p = predict_prob_from_logits(logits)
    return torch.mean(-pos_weight * y * torch.log(p) - (1.0 - y) * torch.log(1.0 - p))


def optimize_single_v(
    z0_train: torch.Tensor,
    y_train: torch.Tensor,
    max_abs_v: float,
    steps: int,
    lr: float,
    device: str,
    pos_weight: float = 1.0,
    min_steps: int = 10,
    patience: int = 5,
    tol: float = 1e-4,
):
    z0 = z0_train.to(device)
    y = y_train.to(device)

    v = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))
    opt = torch.optim.Adam([v], lr=lr)

    last_loss = None
    no_improve = 0
    for step in range(int(steps)):
        opt.zero_grad()
        v_eff = torch.clamp(v, min=-max_abs_v, max=max_abs_v)
        loss = bce_loss_from_logits(z0 + v_eff, y, pos_weight=pos_weight)
        loss.backward()
        opt.step()

        loss_val = float(loss.item())
        if step + 1 >= int(min_steps):
            if last_loss is not None and (last_loss - loss_val) < float(tol):
                no_improve += 1
            else:
                no_improve = 0
            if no_improve >= int(patience):
                break
        last_loss = loss_val

    with torch.no_grad():
        v_star = float(torch.clamp(v, min=-max_abs_v, max=max_abs_v).item())
        train_loss_at_0 = float(bce_loss_from_logits(z0, y, pos_weight=pos_weight).item())
        train_loss_at_v = float(bce_loss_from_logits(z0 + v_star, y, pos_weight=pos_weight).item())

    return v_star, train_loss_at_0, train_loss_at_v


def estimate_single_v_analytic(
    z0_train: torch.Tensor,
    y_train: torch.Tensor,
    max_abs_v: float,
    steps: int,
    lr: float,
    pos_weight: float,
    min_steps: int,
    patience: int,
    tol: float,
):
    # In the aligned SurprisalAggregator forward pass, synergy is added before softplus.
    # The previous closed form no longer applies, so this mode falls back to the same
    # scalar fit while keeping the CLI surface stable.
    return optimize_single_v(
        z0_train=z0_train,
        y_train=y_train,
        max_abs_v=max_abs_v,
        steps=steps,
        lr=lr,
        device="cpu",
        pos_weight=pos_weight,
        min_steps=min_steps,
        patience=patience,
        tol=tol,
    )


_WORKER_ACTIVE_TRAIN = None
_WORKER_ACTIVE_VALID = None
_WORKER_Z0_TRAIN = None
_WORKER_Y_TRAIN = None
_WORKER_Z0_VALID = None
_WORKER_Y_VALID = None
_WORKER_K_VALID = 0
_WORKER_K_TRAIN = 0
_WORKER_MAX_ABS_V = 7.0
_WORKER_SINGLE_V_STEPS = 60
_WORKER_SINGLE_V_LR = 0.05
_WORKER_SINGLE_V_MIN_STEPS = 5
_WORKER_SINGLE_V_PATIENCE = 3
_WORKER_SINGLE_V_TOL = 1e-4
_WORKER_V_MODE = "analytic"
_WORKER_POS_WEIGHT = 1.0


def _init_candidate_worker(
    active_train,
    active_valid,
    z0_train,
    y_train,
    z0_valid,
    y_valid,
    k_valid,
    k_train,
    max_abs_v,
    single_v_steps,
    single_v_lr,
    single_v_min_steps,
    single_v_patience,
    single_v_tol,
    v_mode,
    pos_weight,
):
    global _WORKER_ACTIVE_TRAIN
    global _WORKER_ACTIVE_VALID
    global _WORKER_Z0_TRAIN
    global _WORKER_Y_TRAIN
    global _WORKER_Z0_VALID
    global _WORKER_Y_VALID
    global _WORKER_K_VALID
    global _WORKER_K_TRAIN
    global _WORKER_MAX_ABS_V
    global _WORKER_SINGLE_V_STEPS
    global _WORKER_SINGLE_V_LR
    global _WORKER_SINGLE_V_MIN_STEPS
    global _WORKER_SINGLE_V_PATIENCE
    global _WORKER_SINGLE_V_TOL
    global _WORKER_V_MODE
    global _WORKER_POS_WEIGHT

    _WORKER_ACTIVE_TRAIN = active_train
    _WORKER_ACTIVE_VALID = active_valid
    _WORKER_Z0_TRAIN = z0_train
    _WORKER_Y_TRAIN = y_train
    _WORKER_Z0_VALID = z0_valid
    _WORKER_Y_VALID = y_valid
    _WORKER_K_VALID = int(k_valid)
    _WORKER_K_TRAIN = int(k_train)
    _WORKER_MAX_ABS_V = float(max_abs_v)
    _WORKER_SINGLE_V_STEPS = int(single_v_steps)
    _WORKER_SINGLE_V_LR = float(single_v_lr)
    _WORKER_SINGLE_V_MIN_STEPS = int(single_v_min_steps)
    _WORKER_SINGLE_V_PATIENCE = int(single_v_patience)
    _WORKER_SINGLE_V_TOL = float(single_v_tol)
    _WORKER_V_MODE = str(v_mode)
    _WORKER_POS_WEIGHT = float(pos_weight)


def _evaluate_candidate_job(job):
    idx, a, b, lift, supp_file = job

    valid_mask = torch.tensor([(a in rs) and (b in rs) for rs in _WORKER_ACTIVE_VALID], dtype=torch.bool)
    n_valid = int(valid_mask.sum().item())
    if n_valid < _WORKER_K_VALID:
        return {"status": "skip_valid_small"}

    train_mask = torch.tensor([(a in rs) and (b in rs) for rs in _WORKER_ACTIVE_TRAIN], dtype=torch.bool)
    n_train = int(train_mask.sum().item())
    if n_train < _WORKER_K_TRAIN:
        return {"status": "skip_train_small"}

    z0_sub = _WORKER_Z0_TRAIN[train_mask]
    y_sub = _WORKER_Y_TRAIN[train_mask]
    if _WORKER_V_MODE == "analytic":
        v_star, train_l0, train_lv = estimate_single_v_analytic(
            z0_train=z0_sub,
            y_train=y_sub,
            max_abs_v=_WORKER_MAX_ABS_V,
            steps=_WORKER_SINGLE_V_STEPS,
            lr=_WORKER_SINGLE_V_LR,
            pos_weight=_WORKER_POS_WEIGHT,
            min_steps=_WORKER_SINGLE_V_MIN_STEPS,
            patience=_WORKER_SINGLE_V_PATIENCE,
            tol=_WORKER_SINGLE_V_TOL,
        )
    else:
        v_star, train_l0, train_lv = optimize_single_v(
            z0_train=z0_sub,
            y_train=y_sub,
            max_abs_v=_WORKER_MAX_ABS_V,
            steps=_WORKER_SINGLE_V_STEPS,
            lr=_WORKER_SINGLE_V_LR,
            device="cpu",
            pos_weight=_WORKER_POS_WEIGHT,
            min_steps=_WORKER_SINGLE_V_MIN_STEPS,
            patience=_WORKER_SINGLE_V_PATIENCE,
            tol=_WORKER_SINGLE_V_TOL,
        )

    with torch.no_grad():
        val_l0 = float(bce_loss_from_logits(_WORKER_Z0_VALID[valid_mask], _WORKER_Y_VALID[valid_mask], pos_weight=_WORKER_POS_WEIGHT).item())
        val_lv = float(
            bce_loss_from_logits(_WORKER_Z0_VALID[valid_mask] + float(v_star), _WORKER_Y_VALID[valid_mask], pos_weight=_WORKER_POS_WEIGHT).item()
        )
        delta_val = float(val_l0 - val_lv)

    return {
        "status": "ok",
        "candidate_index": int(idx),
        "rule_a": int(a),
        "rule_b": int(b),
        "lift": float(lift),
        "support_file": int(supp_file),
        "support_train": int(n_train),
        "support_valid": int(n_valid),
        "v_star": float(v_star),
        "train_loss_0": float(train_l0),
        "train_loss_v": float(train_lv),
        "valid_loss_0": float(val_l0),
        "valid_loss_v": float(val_lv),
        "valid_gain": float(delta_val),
    }


def _run_candidate_jobs_sequential(jobs):
    rows = []
    skipped_valid_small = 0
    skipped_train_small = 0
    for job in tqdm(jobs, total=len(jobs), desc="synergy-fit", leave=False):
        out = _evaluate_candidate_job(job)
        status = out.get("status", "")
        if status == "ok":
            rows.append(out)
        elif status == "skip_valid_small":
            skipped_valid_small += 1
        elif status == "skip_train_small":
            skipped_train_small += 1
    return rows, skipped_valid_small, skipped_train_small


def prefilter_candidates_by_valid(active_valid, candidates, min_valid):
    min_valid = int(min_valid)
    if min_valid <= 0:
        return list(range(len(candidates)))
    if len(candidates) == 0 or len(active_valid) == 0:
        return []

    adj = defaultdict(list)
    for idx, (a, b, _lift, _supp) in enumerate(candidates):
        adj[int(a)].append((int(b), idx))

    counts = [0] * len(candidates)
    keep = [False] * len(candidates)
    remaining = len(candidates)

    for rs in tqdm(active_valid, total=len(active_valid), desc="prefilter-valid", leave=False):
        if remaining <= 0:
            break
        if len(rs) < 2:
            continue
        for a in rs:
            if a not in adj:
                continue
            for b, idx in adj[a]:
                if keep[idx]:
                    continue
                if b in rs:
                    counts[idx] += 1
                    if counts[idx] >= min_valid:
                        keep[idx] = True
                        remaining -= 1

    return [i for i, k in enumerate(keep) if k]


def build_indicator_matrix(active_rule_sets, selected_pairs: List[Tuple[int, int]], device: str):
    n = len(active_rule_sets)
    m = len(selected_pairs)
    if m == 0:
        return torch.zeros((n, 0), dtype=torch.float32, device=device)

    mat = torch.zeros((n, m), dtype=torch.float32, device=device)
    for i, rs in enumerate(active_rule_sets):
        for j, (a, b) in enumerate(selected_pairs):
            if (a in rs) and (b in rs):
                mat[i, j] = 1.0
    return mat


class FrozenSynergyModel(nn.Module):
    WEIGHT_MIN = 0.0
    WEIGHT_MAX = 7.0

    def __init__(self, relation_rule_ids, base_weights, bias, selected_pairs, v_init):
        super().__init__()
        self.relation_rule_ids = list(relation_rule_ids)
        self.num_relation_rules = len(self.relation_rule_ids)
        self.pad_local_tok = self.num_relation_rules

        max_global_rule_id = max(self.relation_rule_ids) if self.relation_rule_ids else 0
        global_to_local = torch.full((max_global_rule_id + 1,), self.pad_local_tok, dtype=torch.long)
        if self.num_relation_rules > 0:
            global_to_local[torch.tensor(self.relation_rule_ids, dtype=torch.long)] = torch.arange(
                self.num_relation_rules, dtype=torch.long
            )
        self.register_buffer("global_to_local", global_to_local)
        self.register_buffer("rule_weights", torch.as_tensor(base_weights, dtype=torch.float32).reshape(-1))
        self.register_buffer("bias", torch.tensor(float(bias), dtype=torch.float32))

        local_pairs = []
        global_pairs_filtered = []
        for a, b in selected_pairs:
            if a >= self.global_to_local.shape[0] or b >= self.global_to_local.shape[0]:
                continue
            local_a = int(self.global_to_local[a].item())
            local_b = int(self.global_to_local[b].item())
            if local_a == self.pad_local_tok or local_b == self.pad_local_tok:
                continue
            local_pairs.append((local_a, local_b))
            global_pairs_filtered.append((int(a), int(b)))

        self.selected_pairs_global = global_pairs_filtered
        self.num_relation_synergy = len(local_pairs)
        if self.num_relation_synergy > 0:
            self.register_buffer("synergy_pair_a_local", torch.tensor([p[0] for p in local_pairs], dtype=torch.long))
            self.register_buffer("synergy_pair_b_local", torch.tensor([p[1] for p in local_pairs], dtype=torch.long))
            init = torch.zeros((self.num_relation_synergy,), dtype=torch.float32)
            if v_init.numel() > 0:
                init[: min(self.num_relation_synergy, int(v_init.numel()))] = v_init[: self.num_relation_synergy].float()
        else:
            self.register_buffer("synergy_pair_a_local", torch.empty((0,), dtype=torch.long))
            self.register_buffer("synergy_pair_b_local", torch.empty((0,), dtype=torch.long))
            init = torch.empty((0,), dtype=torch.float32)
        self.synergy = nn.Parameter(init)

    def forward(self, rules):
        if rules.numel() == 0:
            return torch.empty((rules.shape[0],), dtype=torch.float32, device=rules.device)

        rules = rules.long()
        local_rules = torch.full_like(rules, self.pad_local_tok)
        valid = (rules >= 0) & (rules < self.global_to_local.shape[0])
        local_rules[valid] = self.global_to_local[rules[valid]]
        mask = local_rules == self.pad_local_tok

        score = torch.zeros((rules.shape[0],), dtype=torch.float32, device=rules.device)
        if self.num_relation_rules > 0:
            local_safe = local_rules.clone()
            local_safe[mask] = 0
            rule_w = self.rule_weights[local_safe]
            rule_w = torch.clamp(rule_w, min=self.WEIGHT_MIN, max=self.WEIGHT_MAX)
            rule_w.masked_fill_(mask, 0.0)
            score = rule_w.sum(dim=1)

        if self.num_relation_synergy > 0:
            batch_size = int(local_rules.shape[0])
            active = ~mask
            active_matrix = torch.zeros((batch_size, self.num_relation_rules), dtype=torch.bool, device=rules.device)
            row_idx = torch.arange(batch_size, device=rules.device).unsqueeze(1).expand_as(local_rules)
            active_matrix[row_idx[active], local_rules[active]] = True
            synergy_w = torch.clamp(self.synergy, min=-self.WEIGHT_MAX, max=self.WEIGHT_MAX)
            pair_active = active_matrix[:, self.synergy_pair_a_local] & active_matrix[:, self.synergy_pair_b_local]
            score = score + (pair_active.float() * synergy_w.reshape(1, -1)).sum(dim=1)

        logits = score + self.bias
        return predict_prob_from_logits(logits)


def materialize_compact_split_to_padded(split_dict, pad_tok: int):
    offsets = split_dict["offsets"].long()
    rules_flat = split_dict["rules_flat"].int()
    ys = split_dict["golds"].float()

    num_samples = int(ys.shape[0])
    if num_samples == 0:
        return torch.empty((0, 0), dtype=torch.int32), ys

    lengths = offsets[1:] - offsets[:-1]
    max_len = int(lengths.max().item())
    padded = torch.full((num_samples, max_len), pad_tok, dtype=torch.int32)
    for i in range(num_samples):
        start = int(offsets[i].item())
        end = int(offsets[i + 1].item())
        n = end - start
        if n > 0:
            padded[i, :n] = rules_flat[start:end]
    return padded, ys


def BCELossR(weights=[1, 1], reduction="mean"):
    def loss(input, target):
        input = torch.clamp(input, min=1e-7, max=1 - 1e-7)
        bce = -weights[1] * target * torch.log(input) - (1 - target) * weights[0] * torch.log(1 - input)
        if reduction == "sum":
            return torch.sum(bce)
        return torch.mean(bce)
    return loss


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

    pairwise_cmp = base_scores.unsqueeze(0)
    gold_scores_col = gold_scores.unsqueeze(1)
    n_less = (pairwise_cmp < gold_scores_col).sum(dim=1).float()
    n_equal = (pairwise_cmp == gold_scores_col).sum(dim=1).float()

    fill_t = torch.tensor(fill_value, device=scores_tensor.device)
    n_less = n_less - (fill_t < gold_scores).float()
    n_equal = n_equal + 1.0 - (fill_t == gold_scores).float()
    return n_less + (n_equal + 1.0) / 2.0


def rank_batch_group(nnm, batch_items, pad_tok):
    model_device = next(nnm.parameters()).device
    outputs = [None] * len(batch_items)
    fill_value = 0.0

    for pos, (golds_t, candidates_t, rules_t, test_filter_t) in enumerate(batch_items):
        n = len(golds_t)
        if len(candidates_t) == 0 or len(rules_t) == 0:
            empty = torch.empty((0,), dtype=torch.float32, device=model_device)
            outputs[pos] = (empty, empty, n)
            continue

        max_len = int(rules_t.shape[1])
        padded = rules_t
        if padded.dtype != torch.long:
            padded = padded.long()
        scores = nnm(padded)
        scores_raw = scores.clone()
        rank = _rank_from_scores_tensor(scores, golds_t, test_filter_t, fill_value=fill_value)
        rank_raw = _rank_from_scores_tensor(scores_raw, golds_t, test_filter_t, fill_value=fill_value)
        outputs[pos] = (rank, rank_raw, n)
    return outputs


def get_ranks(nnm, sp_to_o, processed, relation, direction, eval_keys, other_split_filter, device, pad_tok, key_batch_size):
    nnm.eval()
    keys = eval_keys.get(relation, [])
    if len(keys) == 0:
        empty = torch.empty((0,), dtype=torch.float32, device=device)
        return empty, empty, 0

    data = []
    for key in keys:
        test_filter = None
        if other_split_filter is not None and key in other_split_filter:
            test_filter = other_split_filter[key].long().to(device, non_blocking=True)

        golds = sp_to_o[key].long().to(device, non_blocking=True)
        candidates = torch.empty((0,), dtype=torch.long, device=device)
        rules = torch.empty((0, 0), dtype=torch.long, device=device)
        if key in processed:
            if "candidates_tensor_eval" not in processed[key]:
                processed[key]["candidates_tensor_eval"] = torch.as_tensor(
                    processed[key]["candidates"], dtype=torch.long, device=device
                )
            candidates = processed[key]["candidates_tensor_eval"]
            if "rules_padded_tensor_eval" not in processed[key]:
                rule_lists = processed[key]["rules"]
                if len(rule_lists) > 0:
                    processed[key]["rules_padded_tensor_eval"] = torch.nested.to_padded_tensor(
                        torch.nested.nested_tensor([torch.tensor(x) for x in rule_lists]), padding=pad_tok
                    ).long()
                else:
                    processed[key]["rules_padded_tensor_eval"] = torch.empty((0, 0), dtype=torch.long)
            if "rules_padded_tensor_eval_gpu" not in processed[key]:
                processed[key]["rules_padded_tensor_eval_gpu"] = processed[key]["rules_padded_tensor_eval"].to(
                    device, non_blocking=True
                )
            rules = processed[key]["rules_padded_tensor_eval_gpu"]
        data.append((golds, candidates, rules, test_filter))

    results = []
    key_batch_size = max(int(key_batch_size), 1)
    for start in range(0, len(data), key_batch_size):
        group = data[start: start + key_batch_size]
        results.extend(rank_batch_group(nnm, group, pad_tok))

    rank, rank_raw, ns = zip(*results)
    return torch.hstack(rank), torch.hstack(rank_raw), sum(ns)


def calc_metric_set(ranks, n):
    if n == 0:
        return {"mrr": 0.0, "h1": 0.0, "h10": 0.0}
    return {
        "mrr": float(((1.0 / ranks).sum() / n).item()),
        "h1": float(((ranks == 1.0).sum() / n).item()),
        "h10": float(((ranks <= 10.0).sum() / n).item()),
    }


def evaluate_link_prediction_metrics(nnm, relation, split_name, device, pad_tok, key_batch_size):
    if split_name == "valid":
        head_sp_to_o = valid_po_to_s
        head_processed = processed_po_valid
        tail_sp_to_o = valid_sp_to_o
        tail_processed = processed_sp_valid
        head_keys = relation_keys["valid_s"]
        tail_keys = relation_keys["valid_o"]
        head_filter = test_po_to_s
        tail_filter = test_sp_to_o
    else:
        head_sp_to_o = test_po_to_s
        head_processed = processed_po_test
        tail_sp_to_o = test_sp_to_o
        tail_processed = processed_sp_test
        head_keys = relation_keys["test_s"]
        tail_keys = relation_keys["test_o"]
        head_filter = None
        tail_filter = None

    tail_rank, tail_rank_raw, tail_n = get_ranks(
        nnm, tail_sp_to_o, tail_processed, relation, "o", tail_keys, tail_filter, device, pad_tok, key_batch_size
    )
    head_rank, head_rank_raw, head_n = get_ranks(
        nnm, head_sp_to_o, head_processed, relation, "s", head_keys, head_filter, device, pad_tok, key_batch_size
    )

    tail_metrics = calc_metric_set(tail_rank, tail_n)
    tail_metrics_raw = calc_metric_set(tail_rank_raw, tail_n)
    head_metrics = calc_metric_set(head_rank, head_n)
    head_metrics_raw = calc_metric_set(head_rank_raw, head_n)
    return {
        "mrr": float((tail_metrics["mrr"] + head_metrics["mrr"]) / 2.0),
        "h1": float((tail_metrics["h1"] + head_metrics["h1"]) / 2.0),
        "h10": float((tail_metrics["h10"] + head_metrics["h10"]) / 2.0),
        "mrr_raw": float((tail_metrics_raw["mrr"] + head_metrics_raw["mrr"]) / 2.0),
        "h1_raw": float((tail_metrics_raw["h1"] + head_metrics_raw["h1"]) / 2.0),
        "h10_raw": float((tail_metrics_raw["h10"] + head_metrics_raw["h10"]) / 2.0),
    }


def joint_finetune(
    train_rules,
    y_train,
    model,
    valid_rules,
    y_valid,
    max_abs_v,
    pos_weight,
    lr,
    steps,
    batch_size,
):
    if getattr(model, "num_relation_synergy", 0) == 0:
        with torch.no_grad():
            base_train = float(BCELossR([1, pos_weight])(model(train_rules), y_train).item())
            base_valid = float(BCELossR([1, pos_weight])(model(valid_rules), y_valid).item())
        return {
            "train_loss_before": base_train,
            "train_loss_after": base_train,
            "valid_loss_before": base_valid,
            "valid_loss_after": base_valid,
            "valid_gain": 0.0,
            "v_final": [],
        }

    loss_fn = BCELossR([1, pos_weight])
    opt = torch.optim.Adam([model.synergy], lr=lr)

    with torch.no_grad():
        model.synergy.data.clamp_(min=-max_abs_v, max=max_abs_v)
        train_before = float(loss_fn(model(train_rules), y_train).item())
        valid_before = float(loss_fn(model(valid_rules), y_valid).item())

    for _ in range(int(steps)):
        perm = torch.randperm(train_rules.shape[0], device=train_rules.device)
        for start in range(0, train_rules.shape[0], int(batch_size)):
            idx = perm[start: start + int(batch_size)]
            opt.zero_grad()
            loss = loss_fn(model(train_rules[idx]), y_train[idx])
            loss.backward()
            opt.step()
            with torch.no_grad():
                model.synergy.data.clamp_(min=-max_abs_v, max=max_abs_v)

    with torch.no_grad():
        train_after = float(loss_fn(model(train_rules), y_train).item())
        valid_after = float(loss_fn(model(valid_rules), y_valid).item())
        v_final = [float(x) for x in model.synergy.detach().cpu().tolist()]

    return {
        "train_loss_before": train_before,
        "train_loss_after": train_after,
        "valid_loss_before": valid_before,
        "valid_loss_after": valid_after,
        "valid_gain": float(valid_before - valid_after),
        "v_final": v_final,
    }


def get_parser():
    parser = argparse.ArgumentParser(description="Incremental Synergy Selection for rule aggregation models")
    parser.add_argument("-d", "--dataset", default="codex-m")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("-r", "--relation", type=int, required=True, help="Relation id; use -1 to iterate all relations")
    parser.add_argument(
        "--experiment", help="Experiment folder name or path", default="exp-1_SurprisalAggregator_1_0"
    )

    parser.add_argument("--min_synergy", type=float, default=0.01)
    parser.add_argument("--candidate_min_support", type=int, default=5, help="Filter candidate list by support in synergy file")
    parser.add_argument("--k_valid", type=int, default=5, help="Minimum |D_e^val|")
    parser.add_argument("--k_train", type=int, default=5, help="Minimum |D_e^train|")

    parser.add_argument("--single_v_steps", type=int, default=30)
    parser.add_argument("--single_v_lr", type=float, default=0.05)
    parser.add_argument("--single_v_min_steps", type=int, default=5)
    parser.add_argument("--single_v_patience", type=int, default=3)
    parser.add_argument("--single_v_tol", type=float, default=1e-4)
    parser.add_argument("--max_abs_v", type=float, default=7.0)
    parser.add_argument(
        "--v_mode",
        type=str,
        default="analytic",
        choices=["analytic", "train"],
        help="Per-candidate v estimation mode: analytic (fast, no training) or train (Adam)",
    )
    parser.add_argument(
        "--min_valid_gain",
        type=float,
        default=0.001,
        help="Keep candidates with valid_gain strictly larger than this threshold",
    )

    parser.add_argument("--joint_steps", type=int, default=300)
    parser.add_argument("--joint_lr", type=float, default=0.02)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--pos", type=float, default=5.0, help="Positive-example loss weight, aligned with aggregation.py")
    parser.add_argument("--eval_key_batch_size", type=int, default=64)

    parser.add_argument("--device", default="cuda", help="Device for joint fine-tuning and metrics; default cuda")
    return parser

args = get_parser().parse_args()

dataset_dir = os.path.join(args.data_root, args.dataset)
expl_dir = os.path.join(dataset_dir, "expl")
datasets_dir = os.path.join(dataset_dir, "datasets")
if os.path.isabs(args.experiment):
    experiment_dir = args.experiment
else:
    experiment_dir = os.path.join(dataset_dir, args.experiment)
output_dir = experiment_dir + "_synergy"
os.makedirs(output_dir, exist_ok=True)

rule_file = os.path.join(dataset_dir, "rules", "rules-1000")
synergy_file = os.path.join(dataset_dir, "rules", "synergy.txt")

device = args.device
if str(device).startswith("cuda") and (not torch.cuda.is_available()):
    print("[WARN] CUDA unavailable, fallback to CPU")
    device = "cpu"

relation_ids = read_ids(os.path.join(dataset_dir, "relation_ids.del"))
rule_meta = parse_rule_file_metadata(rule_file, relation_ids)
rule_map = rule_meta["rule_map"]
PAD_TOK = int(rule_meta["max_rule_id"]) + 1

# Valid/test splits: load shared objects once; relation-specific compact split is built per relation.
c = kge.Config()
c.set("dataset.name", args.dataset)
dataset = kge.Dataset.create(c)
valid_sp_to_o = dataset.index("valid_sp_to_o")
valid_po_to_s = dataset.index("valid_po_to_s")
test_sp_to_o = dataset.index("test_sp_to_o")
test_po_to_s = dataset.index("test_po_to_s")

processed_sp_valid = load_pickle(os.path.join(expl_dir, "processed_sp_valid.pkl"))
processed_po_valid = load_pickle(os.path.join(expl_dir, "processed_po_valid.pkl"))
processed_sp_test = load_pickle(os.path.join(expl_dir, "processed_sp_test.pkl"))
processed_po_test = load_pickle(os.path.join(expl_dir, "processed_po_test.pkl"))


def build_relation_keys(index_dict, direction):
    out = defaultdict(list)
    for key in index_dict.keys():
        if direction == "o":
            _e, r = key
        else:
            r, _e = key
        out[int(r)].append(key)
    return out


relation_keys = {
    "valid_o": build_relation_keys(valid_sp_to_o, "o"),
    "valid_s": build_relation_keys(valid_po_to_s, "s"),
    "test_o": build_relation_keys(test_sp_to_o, "o"),
    "test_s": build_relation_keys(test_po_to_s, "s"),
}

if int(args.relation) == -1:
    relation_list = sorted(rule_map.keys())
    print(f"[INFO] relation=-1 -> iterate all relations with rules, count={len(relation_list)}")
else:
    relation_list = [int(args.relation)]

all_summaries = []
for relation in relation_list:
    print(f"\n[INFO] ===== relation={relation} =====")

    mrr_pickle = os.path.join(experiment_dir, f"mrr-{relation}.pkl")
    if not os.path.exists(mrr_pickle):
        print(f"[WARN] skip relation={relation}, mrr pickle not found: {mrr_pickle}")
        continue

    relation_rule_ids = sorted(rule_map.get(relation, []))
    if len(relation_rule_ids) == 0:
        print(f"[WARN] skip relation={relation}, no rule IDs found")
        continue

    train_dataset_path = os.path.join(datasets_dir, f"dataset_{relation}.p")
    if not os.path.exists(train_dataset_path):
        print(f"[WARN] skip relation={relation}, dataset not found: {train_dataset_path}")
        continue

    load_args = argparse.Namespace(mrr_pickle=mrr_pickle, checkpoint="")
    base = load_base_params(load_args, relation_rule_ids)
    rule_global_to_local = {rid: i for i, rid in enumerate(relation_rule_ids)}

    print(f"[INFO] relation={relation}, #base_rules={len(relation_rule_ids)}")

    # Train split: directly from preprocessed compact dataset.
    train_obj = load_pickle(train_dataset_path)
    if not (isinstance(train_obj, dict) and train_obj.get("format") == "compact_varlen_int32_v1"):
        print(f"[WARN] skip relation={relation}, dataset format is not compact_varlen_int32_v1")
        continue
    train_split = train_obj["train"]

    valid_o = build_compact_split(valid_sp_to_o, processed_sp_valid, relation, direction="o")
    valid_s = build_compact_split(valid_po_to_s, processed_po_valid, relation, direction="s")
    valid_split = concat_compact_splits(valid_o, valid_s)
    test_o = build_compact_split(test_sp_to_o, processed_sp_test, relation, direction="o")
    test_s = build_compact_split(test_po_to_s, processed_po_test, relation, direction="s")
    test_split = concat_compact_splits(test_o, test_s)

    print(
        f"[INFO] samples: train={train_split['num_samples']}, valid={valid_split['num_samples']}, test={test_split['num_samples']}"
    )

    # Step 2: precompute and cache base logits z0(x) before softplus.
    train_cache = precompute_base_logits_and_active_rules(train_split, base, rule_global_to_local)
    valid_cache = precompute_base_logits_and_active_rules(valid_split, base, rule_global_to_local)
    test_cache = precompute_base_logits_and_active_rules(test_split, base, rule_global_to_local)

    z0_cache_path = os.path.join(output_dir, f"z0-cache-r{relation}.pkl")
    save_pickle(
        {
            "train": {"z0": train_cache["z0"], "y": train_cache["y"]},
            "valid": {"z0": valid_cache["z0"], "y": valid_cache["y"]},
            "test": {"z0": test_cache["z0"], "y": test_cache["y"]},
            "relation": int(relation),
        },
        z0_cache_path,
    )
    print(f"[INFO] cached z0 to {z0_cache_path}")

    # Step 3: candidates and activated subsets
    candidates = parse_synergy_file_for_relation(
        synergy_file=synergy_file,
        rule_relation_by_id=rule_meta["rule_relation_by_id"],
        relation=relation,
        min_synergy=args.min_synergy,
        min_supp=args.candidate_min_support,
    )
    print(f"[INFO] loaded candidates={len(candidates)}")

    z0_train = train_cache["z0"]
    y_train = train_cache["y"]
    z0_valid = valid_cache["z0"]
    y_valid = valid_cache["y"]

    active_train = train_cache["active_rule_sets"]
    active_valid = valid_cache["active_rule_sets"]

    prefilter_idx = prefilter_candidates_by_valid(active_valid, candidates, int(args.k_valid))
    if len(prefilter_idx) != len(candidates):
        print(f"[INFO] prefilter kept {len(prefilter_idx)} / {len(candidates)} candidates by valid>= {int(args.k_valid)}")

    rows = []
    skipped_valid_small = 0
    skipped_train_small = 0

    jobs = [(orig_idx, *candidates[orig_idx]) for orig_idx in prefilter_idx]
    if len(jobs) > 0:
        print(f"[INFO] candidate fitting mode=sequential")
        # Always initialize worker globals in parent so sequential path and fork path share logic.
        _init_candidate_worker(
            active_train,
            active_valid,
            z0_train,
            y_train,
            z0_valid,
            y_valid,
            int(args.k_valid),
            int(args.k_train),
            float(args.max_abs_v),
            int(args.single_v_steps),
            float(args.single_v_lr),
            int(args.single_v_min_steps),
            int(args.single_v_patience),
            float(args.single_v_tol),
            str(args.v_mode),
            float(args.pos),
        )
        rows, skipped_valid_small, skipped_train_small = _run_candidate_jobs_sequential(jobs)

    rows.sort(key=lambda x: x["valid_gain"], reverse=True)

    ranked_path = os.path.join(output_dir, f"candidate-ranking-r{relation}.csv")
    with open(ranked_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "candidate_index", "rule_a", "rule_b", "lift", "support_file", "support_train", "support_valid",
            "v_star", "train_loss_0", "train_loss_v", "valid_loss_0", "valid_loss_v", "valid_gain"
        ])
        writer.writeheader()
        if rows:
            writer.writerows(rows)

    # Step 6: keep all improved candidates
    selected = [r for r in rows if float(r["valid_gain"]) > float(args.min_valid_gain)]

    # Step 7: final joint fine-tuning on full train set with base frozen
    selected_pairs = [(int(r["rule_a"]), int(r["rule_b"])) for r in selected]
    selected_v_init = torch.tensor([float(r["v_star"]) for r in selected], dtype=torch.float32)

    train_rules, y_train_padded = materialize_compact_split_to_padded(train_split, PAD_TOK)
    valid_rules, y_valid_padded = materialize_compact_split_to_padded(valid_split, PAD_TOK)
    train_rules_d = train_rules.to(device)
    y_train_d = y_train_padded.reshape(-1).to(device)
    valid_rules_d = valid_rules.to(device)
    y_valid_d = y_valid_padded.reshape(-1).to(device)

    base_model = FrozenSynergyModel(
        relation_rule_ids=relation_rule_ids,
        base_weights=base.local_weights,
        bias=base.bias,
        selected_pairs=[],
        v_init=torch.empty((0,), dtype=torch.float32),
    ).to(device)
    independent_model = FrozenSynergyModel(
        relation_rule_ids=relation_rule_ids,
        base_weights=base.local_weights,
        bias=base.bias,
        selected_pairs=selected_pairs,
        v_init=selected_v_init,
    ).to(device)
    joint_model = FrozenSynergyModel(
        relation_rule_ids=relation_rule_ids,
        base_weights=base.local_weights,
        bias=base.bias,
        selected_pairs=selected_pairs,
        v_init=selected_v_init,
    ).to(device)

    joint = joint_finetune(
        train_rules=train_rules_d,
        y_train=y_train_d,
        model=joint_model,
        valid_rules=valid_rules_d,
        y_valid=y_valid_d,
        max_abs_v=float(args.max_abs_v),
        pos_weight=float(args.pos),
        lr=float(args.joint_lr),
        steps=int(args.joint_steps),
        batch_size=int(args.batch_size),
    )

    selected_out = []
    for i, r in enumerate(selected):
        rr = dict(r)
        rr["v_joint"] = float(joint["v_final"][i]) if i < len(joint["v_final"]) else float(rr["v_star"])
        selected_out.append(rr)

    stage_base = {
        "valid": evaluate_link_prediction_metrics(base_model, relation, "valid", device, PAD_TOK, args.eval_key_batch_size),
        "test": evaluate_link_prediction_metrics(base_model, relation, "test", device, PAD_TOK, args.eval_key_batch_size),
    }
    stage_independent = {
        "valid": evaluate_link_prediction_metrics(independent_model, relation, "valid", device, PAD_TOK, args.eval_key_batch_size),
        "test": evaluate_link_prediction_metrics(independent_model, relation, "test", device, PAD_TOK, args.eval_key_batch_size),
    }
    stage_joint = {
        "valid": evaluate_link_prediction_metrics(joint_model, relation, "valid", device, PAD_TOK, args.eval_key_batch_size),
        "test": evaluate_link_prediction_metrics(joint_model, relation, "test", device, PAD_TOK, args.eval_key_batch_size),
    }

    metric_payload = {
        "relation": int(relation),
        "dataset": args.dataset,
        "experiment": experiment_dir,
        "output_dir": output_dir,
        "stages": {
            "base_no_synergy": stage_base,
            "selected_independent": stage_independent,
            "joint_finetuned": stage_joint,
        },
        "valid": stage_joint["valid"],
        "test": stage_joint["test"],
    }
    metric_path = os.path.join(output_dir, f"metric-{relation}.json")
    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(metric_payload, f, indent=2)

    summary = {
        "dataset": args.dataset,
        "relation": int(relation),
        "base_model": {
            "mrr_pickle": mrr_pickle,
            "num_base_rules": int(len(relation_rule_ids)),
        },
        "candidates": {
            "loaded": int(len(candidates)),
            "kept_after_filter": int(len(rows)),
            "skipped_valid_small": int(skipped_valid_small),
            "skipped_train_small": int(skipped_train_small),
        },
        "selection": {
            "criterion": f"valid_gain > {float(args.min_valid_gain)}",
            "selected": int(len(selected_out)),
            "v_mode": str(args.v_mode),
        },
        "joint_finetune": joint,
        "metrics": {
            "base_valid_mrr": stage_base["valid"]["mrr"],
            "independent_valid_mrr": stage_independent["valid"]["mrr"],
            "joint_valid_mrr": stage_joint["valid"]["mrr"],
            "joint_test_mrr": stage_joint["test"]["mrr"],
        },
        "paths": {
            "z0_cache": z0_cache_path,
            "candidate_ranking_csv": ranked_path,
            "metric_json": metric_path,
        },
    }

    summary_path = os.path.join(output_dir, f"incremental-synergy-summary-r{relation}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "selected": selected_out}, f, indent=2)

    all_summaries.append(summary)
    print(json.dumps(summary, indent=2))
    print(f"[INFO] summary saved to {summary_path}")

if int(args.relation) == -1:
    batch_summary_path = os.path.join(output_dir, "incremental-synergy-summary-all.json")
    with open(batch_summary_path, "w", encoding="utf-8") as f:
        json.dump({"summaries": all_summaries}, f, indent=2)
    print(f"[INFO] all-relations summary saved to {batch_summary_path}")
