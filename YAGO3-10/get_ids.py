import argparse
import html
import os
from typing import Dict, List, Tuple


def read_txt(triples_path: str, separator: str = "\t") -> List[Tuple[str, str, str]]:
    triples = []
    with open(triples_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            line = html.unescape(line)
            h, r, t = line.split(separator)
            triples.append((h, r, t))
    return triples


def read_map(path: str) -> Dict[str, int]:
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx_str, token = line.split("\t", 1)
            mapping[token] = int(idx_str)
    return mapping


def write_map(path: str, mapping: Dict[str, int]) -> None:
    items = sorted(mapping.items(), key=lambda x: x[1])
    with open(path, "w", encoding="utf-8") as f:
        for token, idx in items:
            f.write(f"{idx}\t{token}\n")


def build_maps_from_splits(input_dir: str, splits: List[str]) -> Tuple[Dict[str, int], Dict[str, int]]:
    entity_set = set()
    relation_set = set()
    for split in splits:
        triples = read_txt(os.path.join(input_dir, f"{split}.txt"))
        for h, r, t in triples:
            entity_set.add(h)
            entity_set.add(t)
            relation_set.add(r)

    # 为了可复现，使用排序后分配 id
    entity_to_id = {ent: idx for idx, ent in enumerate(sorted(entity_set))}
    relation_to_id = {rel: idx for idx, rel in enumerate(sorted(relation_set))}
    return entity_to_id, relation_to_id


def write_del_split(
    input_txt_path: str,
    output_del_path: str,
    entity_to_id: Dict[str, int],
    relation_to_id: Dict[str, int],
) -> None:
    triples = read_txt(input_txt_path)
    with open(output_del_path, "w", encoding="utf-8") as f:
        for h, r, t in triples:
            if h not in entity_to_id or t not in entity_to_id or r not in relation_to_id:
                raise KeyError(
                    f"Unseen token when writing {output_del_path}: "
                    f"h={h in entity_to_id}, r={r in relation_to_id}, t={t in entity_to_id}"
                )
            f.write(f"{entity_to_id[h]}\t{relation_to_id[r]}\t{entity_to_id[t]}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate entity/relation ID maps and train/valid/test .del files for KGE"
    )
    parser.add_argument("--input_dir", default=".", help="Directory that contains train.txt/valid.txt/test.txt")
    parser.add_argument(
        "--rebuild_ids",
        action="store_true",
        help="Rebuild entity_ids.del and relation_ids.del from train/valid/test txt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)

    entity_file = os.path.join(input_dir, "entity_ids.del")
    relation_file = os.path.join(input_dir, "relation_ids.del")
    splits = ["train", "valid", "test"]

    if args.rebuild_ids or not (os.path.exists(entity_file) and os.path.exists(relation_file)):
        entity_to_id, relation_to_id = build_maps_from_splits(input_dir, splits)
        write_map(entity_file, entity_to_id)
        write_map(relation_file, relation_to_id)
        print(f"Rebuilt ID maps: entities={len(entity_to_id)}, relations={len(relation_to_id)}")
    else:
        entity_to_id = read_map(entity_file)
        relation_to_id = read_map(relation_file)
        print(f"Loaded ID maps: entities={len(entity_to_id)}, relations={len(relation_to_id)}")

    for split in splits:
        txt_path = os.path.join(input_dir, f"{split}.txt")
        del_path = os.path.join(input_dir, f"{split}.del")
        write_del_split(txt_path, del_path, entity_to_id, relation_to_id)
        print(f"Wrote {del_path}")


if __name__ == "__main__":
    main()