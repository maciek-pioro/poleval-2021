# usage="bash ${0} data_save_path"
# download models
directory=$(pwd)
cd $1

mkdir models
cd models
wget -nc https://nlp.cs.washington.edu/xorqa/cora/models/all_w100.tsv
wget -nc https://nlp.cs.washington.edu/xorqa/cora/models/mGEN_model.zip
wget -nc https://nlp.cs.washington.edu/xorqa/cora/models/mDPR_biencoder_best.cpt
unzip mGEN_model.zip

mkdir embeddings
cd embeddings
for i in 0 1 2 3 4 5 6 7;
do 
  wget -nc https://nlp.cs.washington.edu/xorqa/cora/models/wikipedia_split/wiki_emb_en_$i 
done
for i in 0 1 2 3 4 5 6 7;
do 
  wget -nc https://nlp.cs.washington.edu/xorqa/cora/models/wikipedia_split/wiki_emb_others_$i  
done
cd ../..

# download eval data
mkdir data
cd data
wget -nc https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_dev_full_v1_1.jsonl
wget -nc https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_train_retrieve_eng_span.jsonl
wget -nc https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_train_full.jsonl
cd ..

cd $directory
# Run mDPR
cd mDPR
python dense_retriever.py \
    --model_file ${1}/models/mDPR_biencoder_best.cpt \
    --ctx_file ${1}/models/all_w100.tsv \
    --qa_file ${1}/data/xor_dev_full_v1_1.jsonl \
    --encoded_ctx_file ${1}"/models/embeddings/wiki_emb_*" \
    --out_file xor_dev_dpr_retrieval_results.json \
    --n-docs 20 --validation_workers 1 --batch_size 256 --add_lang
cd ..

# Convert data 
cd mGEN
python3 convert_dpr_retrieval_results_to_seq2seq.py \
    --dev_fp ${1}/mDPR/xor_dev_dpr_retrieval_results.json \
    --output_dir xorqa_dev_final_retriever_results \
    --top_n 15 \
    --add_lang \
    --xor_engspan_train ${1}/data/xor_train_retrieve_eng_span.jsonl \
    --xor_full_train ${1}/data/xor_train_full.jsonl \
    --xor_full_dev ${1}/data/xor_dev_full_v1_1.jsonl

# Run mGEN
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

# Run evaluation
cd eval_scripts
python eval_xor_full.py --data_file ${1}/data/xor_dev_full_v1_1.jsonl --pred_file ../mGEN/xor_dev_final_results.txt --txt_file
