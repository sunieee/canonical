#!/usr/bin/env bash
set -euo pipefail

# conda create -n kge python=3.8 -y
# conda activate kge
# pip install -r requirements.txt
# cd kge
# git checkout a9ecd249ec2d205df59287f64553a1536add4a43
# pip install -e .
# cd ..
# rm -rf kge/data
# ln -s `pwd`/data kge/data

# fb15k-237 wnrr codex-m codex-l YAGO3-10 KG20C



apply_pyclause_on_split() {
    local split="$1"
    local data_dir="$2"
    local out_dir="$3"
    local rules_file="$4"

    PATH_TRAINING="${data_dir}/train.txt"
    FILTER_W_DATA=1
    if [ "$split" = "test" ]; then
        PATH_VALID="${data_dir}/valid.txt"
        PATH_TEST="${data_dir}/test.txt"
    elif [ "$split" = "valid" ]; then
        PATH_VALID="${data_dir}/empty.txt"
        PATH_TEST="${data_dir}/valid.txt"
    else
        PATH_VALID="${data_dir}/empty.txt"
        PATH_TEST="${data_dir}/train.txt"
        FILTER_W_DATA=0
    fi

    local applied_out="${out_dir}/applied_rules_${split}.json"

    python apply_pyclause.py --filter-w-data "$FILTER_W_DATA" \
        --train "$PATH_TRAINING" --valid "$PATH_VALID" --target "$PATH_TEST" \
        --rules "$rules_file" --output "$applied_out" --topk 100 \
        --worker-threads 20 --aggregation maxplus --min-correct-predictions 5 \
        --read-cyclic-rules 1 --read-acyclic1-rules 1 --read-acyclic2-rules 0 \
        --read-zero-rules 0 --read-uxxc-rules 0 --read-uxxd-rules 0
}

process_dataset() {
    local dataset="$1"
    local repo_dir
    repo_dir="$(pwd)"

    echo "=== Processing ${dataset} ==="
    local data_dir="${repo_dir}/data/${dataset}"
    local out_dir="${data_dir}/expl"
    local rules_file="${data_dir}/rules/rules-1000"

    mkdir -p "$out_dir"

    apply_pyclause_on_split train "$data_dir" "$out_dir" "$rules_file"
    apply_pyclause_on_split valid "$data_dir" "$out_dir" "$rules_file"
    apply_pyclause_on_split test "$data_dir" "$out_dir" "$rules_file"

    python create_datasets.py -d "$dataset" --applied_rules "${out_dir}/applied_rules_train.json"

    python process_rules.py --data_dir "$data_dir" --split valid \
        --target_file "${data_dir}/valid.txt" --applied_rules_file "${out_dir}/applied_rules_valid.json" --save_dir "$out_dir"

    python process_rules.py --data_dir "$data_dir" --split test \
        --target_file "${data_dir}/test.txt" --applied_rules_file "${out_dir}/applied_rules_test.json" --save_dir "$out_dir"

    python aggregation.py -d "$dataset" --relation -1 --multiprocess 3
}

for dataset in codex-m; do
    process_dataset "$dataset"
done

