ln -s `pwd`/codex-m `pwd`/kge/data/codex-m
ln -s `pwd`/wnrr `pwd`/kge/data/wnrr
ln -s `pwd`/fb15k-237 `pwd`/kge/data/fb15k-237


conda create -n kge python=3.8 -y
conda activate kge
pip install -r requirements.txt
cd kge
git checkout a9ecd249ec2d205df59287f64553a1536add4a43
pip install -e .
cd ..


export dataset=codex-m

# 用规则生成 explanations
cd explanations
bash process.sh
cd ..

# 构建 canonical 训练数据
python create_datasets.py \
  -d $dataset \
  -e $dataset/expl/explanations-processed/ \
  -o $dataset/datasets


# 训练 canonical 聚合器 + 链接预测评估
python aggregation.py -d $dataset