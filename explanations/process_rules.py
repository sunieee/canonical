import argparse
import copy
import json
import os
import pickle
from tqdm import tqdm


def read_ids(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read().splitlines()
    ids = []
    for line in raw:
        ids.append(line.split("\t")[1])
    return ids


def read_triples(target_file: str):
    triples = []
    with open(target_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                parts = line.split()
            if len(parts) != 3:
                continue
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def preprocess_candidates_from_applied(
    target_file: str,
    entity_ids: list,
    relation_ids: list,
    applied_rules: dict,
):
    processed = {}
    processed_sp = {}
    processed_po = {}

    entity_id_to_idx = {ent: idx for idx, ent in enumerate(entity_ids)}
    relation_id_to_idx = {rel: idx for idx, rel in enumerate(relation_ids)}

    head_applied = applied_rules.get("head", {})
    tail_applied = applied_rules.get("tail", {})

    longest = 0
    for s_raw, p_raw, o_raw in tqdm(read_triples(target_file)):

        if s_raw not in entity_id_to_idx or p_raw not in relation_id_to_idx or o_raw not in entity_id_to_idx:
            continue

        S = entity_id_to_idx[s_raw]
        P = relation_id_to_idx[p_raw]
        O = entity_id_to_idx[o_raw]

        head_candidates_raw = list(head_applied.get(p_raw, {}).get(o_raw, {}).keys())
        tail_candidates_raw = list(tail_applied.get(p_raw, {}).get(s_raw, {}).keys())

        if "me_myself_i" in head_candidates_raw:
            idx = head_candidates_raw.index("me_myself_i")
            head_candidates_raw[idx] = o_raw

        if "me_myself_i" in tail_candidates_raw:
            idx = tail_candidates_raw.index("me_myself_i")
            tail_candidates_raw[idx] = s_raw

        head_rules_map = head_applied.get(p_raw, {}).get(o_raw, {})
        tail_rules_map = tail_applied.get(p_raw, {}).get(s_raw, {})

        rules_heads = []
        filtered_head_candidates_idx = []
        for cand_raw in head_candidates_raw:
            if cand_raw not in entity_id_to_idx:
                continue
            cand_idx = entity_id_to_idx[cand_raw]
            cands = [int(rid) for rid in head_rules_map.get(cand_raw, []) if int(rid) > 0]
            if len(cands) > longest:
                longest = len(cands)
            filtered_head_candidates_idx.append(cand_idx)
            rules_heads.append(cands)

        rules_tails = []
        filtered_tail_candidates_idx = []
        for cand_raw in tail_candidates_raw:
            if cand_raw not in entity_id_to_idx:
                continue
            cand_idx = entity_id_to_idx[cand_raw]
            cands = [int(rid) for rid in tail_rules_map.get(cand_raw, []) if int(rid) > 0]
            if len(cands) > longest:
                longest = len(cands)
            filtered_tail_candidates_idx.append(cand_idx)
            rules_tails.append(cands)

        raw_meta_processed = {
            "heads": {"candidates": filtered_head_candidates_idx, "rules": rules_heads},
            "tails": {"candidates": filtered_tail_candidates_idx, "rules": rules_tails},
        }

        processed[(S, P, O)] = raw_meta_processed

        if (S, P) in processed_sp:
            idx_O = None
            for idx, cand in enumerate(raw_meta_processed["tails"]["candidates"]):
                if cand == O:
                    idx_O = idx
                    break
            if idx_O is not None:
                processed_sp[(S, P)]["candidates"].append(O)
                processed_sp[(S, P)]["rules"].append(copy.copy(raw_meta_processed["tails"]["rules"][idx_O]))
        else:
            processed_sp[(S, P)] = {
                "candidates": copy.deepcopy(raw_meta_processed["tails"]["candidates"]),
                "rules": copy.deepcopy(raw_meta_processed["tails"]["rules"]),
            }

        if (P, O) in processed_po:
            idx_S = None
            for idx, cand in enumerate(raw_meta_processed["heads"]["candidates"]):
                if cand == S:
                    idx_S = idx
                    break
            if idx_S is not None:
                processed_po[(P, O)]["candidates"].append(S)
                processed_po[(P, O)]["rules"].append(copy.copy(raw_meta_processed["heads"]["rules"][idx_S]))
        else:
            processed_po[(P, O)] = {
                "candidates": copy.deepcopy(raw_meta_processed["heads"]["candidates"]),
                "rules": copy.deepcopy(raw_meta_processed["heads"]["rules"]),
            }

    print(f"Longest rule set for a candidate: {longest}")
    return processed, processed_sp, processed_po


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--target_file", required=True)
    parser.add_argument("--applied_rules_file", required=True)
    parser.add_argument("--save_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.split not in ["train", "valid", "test"]:
        raise Exception("split must be one of train/valid/test")

    os.makedirs(args.save_dir, exist_ok=True)

    ent_ids_file = os.path.join(args.data_dir, "entity_ids.del")
    rel_ids_file = os.path.join(args.data_dir, "relation_ids.del")
    entity_ids = read_ids(ent_ids_file)
    relation_ids = read_ids(rel_ids_file)

    with open(args.applied_rules_file, "r", encoding="utf-8") as f:
        applied_rules = json.load(f)

    processed_candidates, processed_sp, processed_po = preprocess_candidates_from_applied(
        target_file=args.target_file,
        entity_ids=entity_ids,
        relation_ids=relation_ids,
        applied_rules=applied_rules,
    )

    save_path = os.path.join(args.save_dir, f"processed_explanations_{args.split}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(processed_candidates, f, protocol=pickle.HIGHEST_PROTOCOL)

    save_path = os.path.join(args.save_dir, f"processed_sp_{args.split}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(processed_sp, f, protocol=pickle.HIGHEST_PROTOCOL)

    save_path = os.path.join(args.save_dir, f"processed_po_{args.split}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(processed_po, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved processed files for split={args.split} in {args.save_dir}")


if __name__ == "__main__":
    main()
