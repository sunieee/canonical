# conda create -n kge python=3.8 -y
# conda activate kge
# pip install -r requirements.txt
# cd kge
# git checkout a9ecd249ec2d205df59287f64553a1536add4a43
# pip install -e .
# cd ..

for dataset in fb15k-237 wnrr codex-m codex-l YAGO3-10 KG20C; do
    echo "Processing $dataset"
    export dataset=$dataset
    # 用规则生成 explanations

    ln -s `pwd`/$dataset `pwd`/kge/data/$dataset
    cd explanations
    bash process.sh
    cd ..
    # python create_explanations.py -d $dataset

    # 构建 canonical 训练数据
    python create_datasets.py -d $dataset

    # 训练 canonical 聚合器 + 链接预测评估
    python aggregation.py -d $dataset --relation -1 --multiprocess 3
done

