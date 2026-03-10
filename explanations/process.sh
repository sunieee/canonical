#!/bin/bash
if [ -z "${dataset+x}" ]; then
  export dataset=codex-m
fi

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_dir="$(cd "$script_dir/.." && pwd)"

data_dir="$repo_dir/${dataset}/"
rule_file="$data_dir/rules/rules-1000"
anyburl_path="$script_dir/AnyBURL/AnyBURL-PT.jar"
# this is the folder where everything is stored, it will be in the dataset folder
output_folder_name="expl/"
# two-step preprocess scripts
process_explanations_path="$script_dir/preprocess_explanations.py"
convert_to_rules_path="$script_dir/convert_to_rules.py"
process_rules_path="$script_dir/process_rules.py"
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


apply_config="config-apply-explanations.properties"

explanations_folder="${output_dir}explanations/"
processed_explanations_folder="${output_dir}explanations-processed/"


if [ ! -d "$explanations_folder" ]; then
    mkdir "$explanations_folder"
fi

run_split() {
  split="$1"

  PATH_TRAINING="${data_dir}train.txt"
  if [ "$split" = "test" ]; then
    PATH_VALID="${data_dir}valid.txt"
    PATH_TEST="${data_dir}test.txt"
  elif [ "$split" = "valid" ]; then
    PATH_VALID="${data_dir}empty.txt"
    PATH_TEST="${data_dir}valid.txt"
  else
    PATH_VALID="${data_dir}empty.txt"
    PATH_TEST="${data_dir}train.txt"
  fi

  PATH_OUTPUT="${explanations_folder}${split}-explanations"
  apply_config="${explanations_folder}config-apply-explanations-${split}.properties"
  PATH_RULE_INDEX="${explanations_folder}rule-index-${split}"

  /bin/cat <<EOM >$apply_config
PATH_TRAINING  = $PATH_TRAINING
PATH_VALID     = $PATH_VALID
PATH_TEST      = $PATH_TEST

PATH_RULES      = $rule_file
PATH_OUTPUT     = $PATH_OUTPUT


# these params will be ignored if you flip to maxplus or maxplus-explanation-stdout
PATH_RULE_INDEX = $PATH_RULE_INDEX
MAX_EXPLANATIONS = 200

# this setting generates the explanations file and the rule index which is the input to the transformer
AGGREGATION_TYPE = maxplus-explanation

# this setting generates the ranking in the standard output of anyburl, which corresponds to the transformer input
# AGGREGATION_TYPE = maxplus-explanation-stdout

# this setting generates ranking of the original anyburl maxplus algorithm
# AGGREGATION_TYPE = maxplus

READ_CYCLIC_RULES = 1
READ_ACYCLIC1_RULES = 1
READ_ACYCLIC2_RULES = 0
READ_ZERO_RULES = 0
READ_THRESHOLD_CORRECT_PREDICTIONS = 5

TOP_K_OUTPUT = 100
WORKER_THREADS = $worker_threads
EOM

  java -cp $anyburl_path de.unima.ki.anyburl.Apply $apply_config
  # python $process_explanations_path --data_dir $data_dir --split $split --explanation_file $PATH_OUTPUT --rules_index_file $PATH_RULE_INDEX --save_dir $processed_explanations_folder
  APPLIED_RULES_FILE="${processed_explanations_folder}applied_rules_${split}.json"
  python $convert_to_rules_path --explanation_file $PATH_OUTPUT --rules_index_file $PATH_RULE_INDEX --rules_file $rule_file --output_file $APPLIED_RULES_FILE
  python $process_rules_path --data_dir $data_dir --split $split --target_file "$PATH_TEST" --applied_rules_file $APPLIED_RULES_FILE --save_dir $processed_explanations_folder
}

for split in train valid test; do
  run_split "$split"
done
# these params will be ignored if you flip to maxplus or maxplus-explanation-stdout

