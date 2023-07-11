# mOKB6: A Multilingual Open Knowledge Base Benchmark

We release a dataset called mOKB6 for multilingual open knowledge base completion task.
mOKB6 has 42K facts in six languages: English, Hindi, Telugu, Spanish, Portuguese, and Chinese.

We also provide the code of our baseline knowledge graph embedding model, which is adapted from the publicly available official code repository ([link](https://github.com/intfloat/SimKGC)) of [Wang et al., 2022](https://aclanthology.org/2022.acl-long.295).

This work appeared at the **ACL 2023 (main)** conference and more details can be found in the paper ([paper link](https://arxiv.org/abs/2211.06959)).
Checkout the video ([link](https://drive.google.com/file/d/1TYx-FABr9QdrAXFHM3gSxHOoc0WrsFcF/view?usp=drive_link)) or poster ([link](https://drive.google.com/file/d/1PXODKMhFKkw3FRNZOn2xToW5mzIbdT72/view?usp=drive_link)) presentation for a brief overview. 

## mOKB6 Dataset
The `./mokb6/mono/` folder contains the mOKB6 dataset, containing six monolingual open KBs in six languages: 
1. English Open KB inside `./mokb6/mono/mono_en`
2. Hindi Open KB inside `./mokb6/mono/mono_hi`
3. Telugu Open KB inside `./mokb6/mono/mono_te`
4. Spanish Open KB inside `./mokb6/mono/mono_es`
5. Portuguese Open KB inside `./mokb6/mono/mono_pt`
6. Chinese Open KB inside `./mokb6/mono/mono_zh`

Each monolingual Open KB's folder contains three files: `train.txt`, `valid.txt`, and `test.txt`.
These files are the train-dev-test splits of the respective language's Open KB, which contain tab-separated Open IE triples of the form (subject, relation, object).

For reproducibility of our results, we provide the translated Open KB facts.
Thus, for each baseline given in Table 3 in the paper, we provide the corresponding dataset inside `./mokb6/` folder.
For e.g., our best baseline (for all languages except English) called Union+Trans is trained using data contained in `./mokb6/union+trans/` for the 5 languages (`./mokb6/union+trans/union+trans_en2hi/` for Hindi).
Whereas the best performing baseline for English called Union can be reproduced using data contained in `./mokb6/union/`.

## Model
We benchmark Knowledge Graph Embedding (KGE) models from available repositories from [CaRe](https://github.com/malllabiisc/CaRE), [GRU-ConvE](https://github.com/vid-koci/KBCtransferlearning), and [SimKGC](https://github.com/intfloat/SimKGC) on mOKB6 dataset.

We provide the code of SimKGC model (adapted from [Wang et al., 2022](https://aclanthology.org/2022.acl-long.295)) as it showed the best performance when compared with the other KGE models. 
We use it (with mBERT initialization) to report the baselines (Table 3 in the paper) and to do an empirical study of the task (e.g., Figure 3 in the paper).

## How to Run
Here, we provide the commands to reproduce the scores for Union+Trans (given in Table 3 in the paper).

### Requirements

* python>=3.7
* torch>=1.6 (for mixed precision training)
* transformers>=4.15

All experiments are run with a single A100 (40GB) GPU.

### Installation

```
conda create --name mokb python=3.7
conda activate mokb
pip install -r requirements.txt
```

### Preprocessing the dataset
Create `data` directory for saving preprocessed data using `mkdir data`.

First, run the below command to process the data of monolingual Open KBs. This is necessary and must to do to have the correct evaluation of any model, i.e., a given model will be evaluated on each language's Open KB (which will not include any translated facts).

```
sh sh_preprocess_mono_okbs.sh
```

Then, set the environment variables `baseline_name` and `baseline_data` to the name of baseline of interest, and to its corresponding data's path, respectively.

```
baseline_data=./mokb6/union+trans/union+trans_en2hi
baseline_name=union+trans_en2hi
```

Run the following commands to preprocess the data and store it in `./data/${baseline_name}/` folder.

```
python convert_format_mokb.py --train ${baseline_data}/train.txt   --val ${baseline_data}/valid.txt --test ${baseline_data}/test.txt  --out_dir ./data/${baseline_name}

python3 preprocess.py --train-path ./data/${baseline_name}/train.txt --valid-path ./data/${baseline_name}/valid.txt --test-path ./data/${baseline_name}/test.txt --task mopenkb
```

### Training
Set the environment variable `batch_size` as per the baseline (e.g., 128 for Mono baseline for all languages except English, and 256 for the remaining baselines).
```
batch_size=256
```

Run the below command to train SimKGC (mBERT) and store its best checkpoint in `./checkpoint/${baseline_name}/` folder.

```
python3 main.py --model-dir ./checkpoint/${baseline_name} --pretrained-model bert-base-multilingual-cased --pooling mean --lr 3e-5 --train-path ./data/${baseline_name}/train.txt.json  --valid-path ./data/${baseline_name}/valid.txt.json  --task mopenkb --batch-size ${batch_size} --print-freq 20 --additive-margin 0.02 --use-amp --use-self-negative --finetune-t --pre-batch 0 --epochs 100 --workers 3 --max-to-keep 0 --patience 10 --seed 2022
```

### Evaluating
To evaluate the model on a given language's Open KB's testset, say `hi`, set the `language` variable
```
language=hi
```

Then, evaluate the model checkpoint `./checkpoint/${baseline_name}/model_best.mdl` using the below command.

```
python3 evaluate.py --task mopenkb --pretrained-model bert-base-multilingual-cased --is-test --eval-model-path ./checkpoint/${baseline_name}/model_best.mdl --train-path data/mono_${language}/train.txt.json  --valid-path data/mono_${language}/test.txt.json
```

## Cite
If you use or extend our work, please cite it:
```
@inproceedings{mittal-etal-2023-mokb6,
    title = "m{OKB}6: {A} {M}ultilingual {O}pen {K}nowledge {B}ase {C}ompletion {B}enchmark",
    author = "Mittal, Shubham  and
      Kolluru, Keshav  and
      Chakrabarti, Soumen  and
      -, Mausam",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-short.19",
    pages = "201--214",
}
```
