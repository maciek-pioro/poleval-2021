## PolEval 2021 submissions

This repository contains winning submissions for [Task 3: Post-correction of OCR results](http://poleval.pl/tasks/task3)
and [Task 4: Question answering challenge](http://poleval.pl/tasks/task4). Both submission rely on fine-tuning the [mT5 
model](https://github.com/google-research/multilingual-t5) on respective tasks.

Solution details are described in the [workshop proceedings](http://poleval.pl/files/poleval2021.pdf)  

---

### Task 3 results

| model    | dev-0  | test-A | test-B |
|:--------:|:------:|:------:|:------:|
| original | 16.550 | 16.527 | 16.543 |
| base     | 4.678  | 4.792  | 4.796  |
| large    | 4.418  | 4.515  | 4.559  |
| XXL      | 3.604  | 3.725  | 3.744  |

---

### Task 4 results

| model | test-B |
|:-----:|:------:|
| base  | 52.12  |
| large | 59.20  |
| XXL   | 71.68  |

---

## Data preparation

### Setup

Common steps for both tasks

1. Install pip requirements

```shell
pip install -r requirements.txt
```

3. Download mT5 vocabulary to repository root

```shell
gsutil cp gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model .
```

3. Prepare GCS bucket for storing training datasets: https://cloud.google.com/storage/docs/creating-buckets
4. Update `gs_base_path` in `config/config.yaml`


### OCR correction

The provided data contains pages of text which are in many instances longer then maximum 
sequence length allowed by the model architecture. To alleviate that the training 
examples are created by aligning and splitting longer input/output pairs. 

1. Pull task repository
```shell
git clone -b secret https://github.com/poleval/2021-ocr-correction.git
```

2. Split examples into chunks to match maximum sequence length
```shell
python3 -m data_preparation.ocr_correction.split_text \
  2021-ocr-correction \
  --length-limit 384
```

3. Upload files to created bucket, update or match paths from `config/task/ocr_correction.yaml`. 
Keep `.index` files to restore full text from predictions

### Question answering

For question answering the model input prompt consists of question and context passages 
retrieved from Wikipedia. This section shows how to reproduce the data used in submission.

The prepared data is available [here](https://drive.google.com/drive/folders/1g_EV1WBhH8kxZdMLXLVWXQ5HnOeeRZkn?usp=sharing).
Skip to step 5 if using this dataset.


1. Pull task repository
```shell
git clone -b secret https://github.com/poleval/2021-question-answering.git
```

2. Start local Elasticsearch instance using docker (skip if using existing cluster)
```shell
docker volume create poleval-es # recommended for persistence
docker run \
  -p 9200:9200 \
  -p 9300:9300 \
  -v poleval-es:/usr/share/elasticsearch/data \
  -e "discovery.type=single-node" \
  docker.elastic.co/elasticsearch/elasticsearch:7.13.4
```

3. Download spaCy model
```shell
python -m spacy download pl_core_news_md
```

4. Index and retrieve context passages for Polish QA dataset
```shell
python3 -m data_preparation.question_answering.quiz_pl \
  2021-question-answering \
  wiki_passages_pl
```

4. Index and retrieve context passages for TriviaQA dataset
```shell
python3 -m data_preparation.question_answering.trivia_qa wiki_passages_en
```

5. Select questions only for prediction
```shell
cat test-B-input-510.tsv | cut -f1 > test-B-questions-510.tsv
```

6. Upload files to created bucket, update or match paths from `config/task/question_answering.yaml`


## Training and evaluation

The models were trained using TPUv3 device. Model configuration is defined in `config/` folder.
After completing the training inference will be run using prompts from files specified under 
`config/task/<task.yaml> -> predict_files`

1. Start TPUv3 and cloud instance eg. using [ctpu tool](https://github.com/tensorflow/tpu/tree/master/tools/ctpu)

```
ctpu up --name poleval --tpu-size=v3-8 -tf-version 2.5.0 
```

2. SSH to TPU instance, download this repository and install the requirements
3. Start the training (or resume from the latest checkpoint) specifying task and model configuration
```shell
python3 main.py model=xxl task=question_answering +tpu_name=poleval 
```

4. (OCR only) Concatenate the corrected fragments to produce source text

```shell
python3 -m data_preparation.ocr_correction.restore \
  gs://my-bucket/data/ocr/dev-0-input-384.txt-1100000 \
  dev-0-384.index \
  dev-0-restored.txt
```

5. Evaluate results using [geval](https://gitlab.com/filipg/geval) tool

```shell
cd 2021-question-answering # or 2021-ocr-correction
gsutil cp gs://my-bucket/data/polish_qa/test-B-questions-510.tsv-1010000 test-B/out.tsv
./geval --test-name test-B 
```



## Acknowledgments

Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC)
=======
# CORA
This is the official implementation of the following paper:
Akari Asai, Xinyan Yu, Jungo Kasai and Hannaneh Hajishirzi. [One Question Answering Model for Many Languages with Cross-lingual Dense Passage Retrieval](https://arxiv.org/abs/2107.11976). *NeurIPS*. 2021. 

**Important Notice: Recently, we have worked on improving CORA code & documentations for our NAACL 2022 shared task baselies, which can be donwloaded [here](https://github.com/mia-workshop/MIA-Shared-Task-2022). We are planning to update the CORA repository as well, but if you are stating using CORA, potentially using the MIA-Shared-Task baseline codes instead help.**

![cora_image](fig/overview_v4.png)

In this paper, we introduce **CORA**, a single, unified multilingual open QA model for many languages.      
CORA consists of two main components: **mDPR** and **mGEN**.     
**mDPR** retrieves documents from multilingual document collections and **mGEN** generates the answer in the target languages directly instead of using any external machine translation or language-specific retrieval module.      
Our experimental results show state-of-the-art results across two multilingual open QA dataset: [XOR QA](https://nlp.cs.washington.edu/xorqa/) and [MKQA](https://github.com/apple/ml-mkqa). 



##  Contents
0. [Quick Run on XOR QA](#quick-run-on-xor-qa)
1. [Overview](#overview)
2. [Data](#data)
3. [Installation](#installation)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Citations and Contact](#citations-and-contact)

## Quick Run on XOR QA
We provide [quick_start_xorqa.sh](quick_start_xorqa.sh), with which you can easily set up and run evaluation on the XOR QA full dev set.

The script will
1. download our trained mDPR, mGEN and encoded Wikipedia embeddings,
2. run the whole pipeline on the evaluation set, and 
3. calculate the QA scores. 

You can download the prediction results from [here](https://drive.google.com/file/d/1lokVB9ngbOeD26PFBL_4O3h1E4B0saN-/view?usp=sharing).

## Overview
To run **CORA**, you first need to preprocess Wikipedia using the codes in [wikipedia_preprocess](wikipedia_preprocess).      
Then you train [mDPR](mDPR) and [mGEN](mGEN).      
Once you finish training those components, please run evaluations, and then evaluate the performance using [eval_scripts](eval_scripts).

Please see the details of each components in each directory. 

- [mDPR](mDPR): codes for training and evaluating our mDPR.
- [mGEN](mGEN): codes for training and evaluating our mGEN.
- [wikipedia_preprocess](wikipedia_preprocess): codes for preprocessing Wikipedias.
- [eval_scripts](eval_scripts): scripts to evaluate the performance. 


## Data
### Training data
You can download the mDPR train data from [here](https://drive.google.com/drive/folders/17xXq9hnUdpnE6XDHDdr6Wdv0tvOGtS4i?usp=sharing).
- [Initial train data](https://drive.google.com/file/d/1kyitGIjkrAgjTmMaC23pp6C_VIWn0v8M/view?usp=sharing)
- [Train data after the first iteration](https://drive.google.com/file/d/11Za61llqxstKhESrSQww-1WqBDNz_OqU/view?usp=sharing)
- [Final train data (after two iterations)](https://drive.google.com/file/d/1YyG1VjuPN2fh7J_tHEzR1-XDTyEC9cZY/view?usp=sharing)

### Evaluation data
We evaluate our models performance on XOR QA and MKQA. 

- XOR QA 
Please download the XOR QA (full) data by running the command below.

```sh
mkdir data
cd data
wget https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_dev_full_v1_1.jsonl
wget https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_test_full_q_only_v1_1.jsonl
cd ..
```

- MKQA
Please download the original MKQA data from the original repository.

```sh
wget https://github.com/apple/ml-mkqa/raw/master/dataset/mkqa.jsonl.gz
gunzip mkqa.jsonl.gz
```
Before evaluating on MKQA, you need to preprocess the MKQA data to convert them into the same format as XOR QA. Please follow the instructions at [eval_scripts/README.md](eval_scripts/README.md).


## Installation

### Dependencies 
- Python 3
- [PyTorch](https://pytorch.org/) (currently tested on version 1.7.0)
- [Transformers](https://github.com/huggingface/transformers) (version 4.2.1; unlikely to work with a different version)

### Trained models
You can download trained models by running the commands below:
```sh
mkdir models
wget https://nlp.cs.washington.edu/xorqa/cora/models/all_w100.tsv
wget https://nlp.cs.washington.edu/xorqa/cora/models/mGEN_model.zip
wget https://nlp.cs.washington.edu/xorqa/cora/models/mDPR_biencoder_best.cpt
unzip mGEN_model.zip
mkdir embeddings
cd embeddings
for i in 0 1 2 3 4 5 6 7;
do 
  wget https://nlp.cs.washington.edu/xorqa/cora/models/wikipedia_split/wiki_emb_en_$i 
done
for i in 0 1 2 3 4 5 6 7;
do 
  wget https://nlp.cs.washington.edu/xorqa/cora/models/wikipedia_split/wiki_emb_others_$i  
done
cd ../..
```

## Training
CORA is trained with our **iterative training process**, where each iteration proceeds over two states: *parameter updates* and *cross-lingual data expansion*.  

1. Train mDPR with the current training data. For the first iteration, the training data is the gold paragraph data from Natural Questions and TyDi-XOR QA.  
2. Retrieve top documents using trained mDPR 
3. Train mGEN with retrieved data
4. Run mGEN on each passages from mDPR and synthetic data retrieval to label the new training data. 
5. Go back to step 1. 

![overview_training](fig/overview_training.jpg)

See the details of each training step in [mDPR/README.md](mdpr/README.md) and [mGEN/README.md](mGEN/README.md).

## Evaluation

1. Run mDPR on the input data

```sh
python dense_retriever.py \
    --model_file ../models/mDPR_biencoder_best.cpt \
    --ctx_file ../models/all_w100.tsv \
    --qa_file ../data/xor_dev_full_v1_1.jsonl \
    --encoded_ctx_file "../models/embeddings/wiki_emb_*" \
    --out_file xor_dev_dpr_retrieval_results.json \
    --n-docs 20 --validation_workers 1 --batch_size 256 --add_lang
```

2. Convert the retrieved results into mGEN input format

```sh
cd mGEN
python3 convert_dpr_retrieval_results_to_seq2seq.py \
    --dev_fp ../mDPR/xor_dev_dpr_retrieval_results.json \
    --output_dir xorqa_dev_final_retriever_results \
    --top_n 15 \
    --add_lang \
    --xor_engspan_train data/xor_train_retrieve_eng_span.jsonl \
    --xor_full_train data/xor_train_full.jsonl \
    --xor_full_dev data/xor_dev_full_v1_1.jsonl
```

3. Run mGEN
```sh
CUDA_VISIBLE_DEVICES=0 python eval_mgen.py \
    --model_name_or_path \
    --evaluation_set xorqa_dev_final_retriever_results/val.source \
    --gold_data_path xorqa_dev_final_retriever_results/gold_para_qa_data_dev.tsv \
    --predictions_path xor_dev_final_results.txt \
    --gold_data_mode qa \
    --model_type mt5 \
    --max_length 20 \
    --eval_batch_size 4
cd ..
```

4. Run the XOR QA full evaluation script
```sh
cd eval_scripts
python eval_xor_full.py --data_file ../data/xor_dev_full_v1_1.jsonl --pred_file ../mGEN/xor_dev_final_results.txt --txt_file
```

## Baselines
In our paper, we have tested several baselines such as Translate-test or multilingual baselines. The codes for machine translations or BM 25-based retrievers are at [baselines](baselines). To run the baselines, you may need to download code and mdoels from [the XOR QA repository](https://github.com/AkariAsai/XORQA). Those codes are implemented by [Velocity](https://github.com/velocityCavalry) :)


## Citations and Contact

If you find this codebase is useful or use in your work, please cite our paper.
```
@inproceedings{
asai2021cora,
title={One Question Answering Model for Many Languages with Cross-lingual Dense Passage Retrieval},
author={Akari Asai and Xinyan Yu and Jungo Kasai and Hannaneh Hajishirzi},
booktitle={NeurIPS},
year={2021}
}
```
Please contact Akari Asai ([@AkariAsai](https://twitter.com/AkariAsai) on Twitter, akari[at]cs.washington.edu) for questions and suggestions.

