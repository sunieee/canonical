#!/bin/bash
if [ -z "${dataset+x}" ]; then
  export dataset=codex-m
fi

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_dir="$(cd "$script_dir/.." && pwd)"

data_dir="$repo_dir/${dataset}/"
rule_file="$data_dir/rules/rules-1000"
# this is the folder where everything is stored, it will be in the dataset folder
output_folder_name="expl/"
convert_to_rules_path="$script_dir/convert_to_rules.py"
process_rules_path="$script_dir/process_rules.py"
apply_pyclause_path="$script_dir/apply_pyclause.py"
worker_threads=20

output_dir=$data_dir$output_folder_name

echo "$output_dir"

if [ ! -d "$output_dir" ]; then
    mkdir "$output_dir"
else
  echo "Output folder exists already" 1>&2
  echo "$output_dir"
  #exit 1
fi

if [ ! -f "$rule_file" ]; then
  echo "Found no rules in" 1>&2
  echo "$rule_file"
  exit 1
fi

explanations_folder="${output_dir}explanations/"
processed_explanations_folder="${output_dir}explanations-processed/"

if [ ! -d "$explanations_folder" ]; then
    mkdir "$explanations_folder"
fi

run_split() {
  split="$1"

  PATH_TRAINING="${data_dir}train.txt"
  FILTER_W_DATA=1
  if [ "$split" = "test" ]; then
    PATH_VALID="${data_dir}valid.txt"
    PATH_TEST="${data_dir}test.txt"
  elif [ "$split" = "valid" ]; then
    PATH_VALID="${data_dir}empty.txt"
    PATH_TEST="${data_dir}valid.txt"
  else
    PATH_VALID="${data_dir}empty.txt"
    PATH_TEST="${data_dir}train.txt"
    FILTER_W_DATA=0
  fi

  APPLIED_RULES_FILE="${processed_explanations_folder}applied_rules_${split}.json"

  python "$apply_pyclause_path"  --train "$PATH_TRAINING"  --valid "$PATH_VALID"  --target "$PATH_TEST" --rules "$rule_file" \
    --output "$APPLIED_RULES_FILE" \
    --topk 100 \
    --filter-w-data "$FILTER_W_DATA" \
    --worker-threads $worker_threads \
    --aggregation maxplus \
    --min-correct-predictions 5 \
    --read-cyclic-rules 1 \
    --read-acyclic1-rules 1 \
    --read-acyclic2-rules 0 \
    --read-zero-rules 0 \
    --read-uxxc-rules 0 \
    --read-uxxd-rules 0
  python $process_rules_path --data_dir $data_dir --split $split --target_file "$PATH_TEST" --applied_rules_file $APPLIED_RULES_FILE --save_dir $processed_explanations_folder
}

for split in train valid test; do
  run_split "$split"
done
