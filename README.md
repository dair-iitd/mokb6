# mOKB6: A Multilingual Open Knowledge Base Benchmark

We release mOKB6, a dataset for multilingual open knowledge base completion task along with the first baseline results for this task.
mOKB6 has 42K facts in six languages: English, Hindi, Telugu, Spanish, Portuguese, and Chinese.

We provide the code of our baseline model, SimKGC, which is adapted from the publicly available official code repository ([link](https://github.com/intfloat/SimKGC)) of [Wang et al., 2022](https://arxiv.org/pdf/2203.02167).

## How to Run

### Requirements

* python>=3.7
* torch>=1.6 (for mixed precision training)
* transformers>=4.15

All experiments are run with a single A100 (40GB) GPU.

### Step 0: installation

```
conda create --name mokb python=3.7
conda activate mokb
pip install -r requirements.txt
```

### Step 1: preprocessing the dataset
Create `data` directory for saving preprocessed data using `mkdir data`.

```
python convert_format_mokb.py --train ./mokb6/mono_en/train.txt   --val ./mokb6/mono_en/valid.txt --test ./mokb6/mono_en/test.txt  --out_dir ./data/mono_en

python3 preprocess.py --train-path ./data/mono_en/train.txt --valid-path ./data/mono_en/valid.txt --test-path ./data/mono_en/test.txt --task mopenkb
```

### Step 2: training SimKGC
```
python3 main.py --model-dir ./checkpoint/mono_en --pretrained-model bert-base-multilingual-cased --pooling mean --lr 3e-5 --train-path ./data/mono_en/train.txt.json  --valid-path ./data/mono_en/valid.txt.json  --task mopenkb --batch-size 256 --print-freq 20 --additive-margin 0.02 --use-amp --use-self-negative --finetune-t --pre-batch 0 --epochs 100 --workers 3 --max-to-keep 0 --patience 10 --seed 2022
```

### Step 3: evaluating SimKGC
```
python3 evaluate.py --task mopenkb --is-test --eval-model-path checkpoint/mono_en/model_best.mdl --train-path data/mono_en/train.txt.json  --valid-path data/mono_en/test.txt.json
```