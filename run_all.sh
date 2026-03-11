python aggregation.py --relation -1 --dataset codex-m > codex-m/run.log 2>&1
python aggregation.py --relation -1 --dataset fb15k-237 > fb15k-237/run.log 2>&1
python aggregation.py --relation -1 --dataset wnrr > wnrr/run.log 2>&1



export dataset=codex-m
python aggregation.py -d $dataset --relation -1 --multiprocess 3 --model SurprisalAggregator
python aggregation.py -d $dataset --relation -1 --multiprocess 3 --model SurprisalAggregator --synergy
