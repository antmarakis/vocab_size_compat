#!/bin/bash

# args:
# 1 - english vocab size
# 2 - other vocab size
# 3 - other lang
# note that for chinese it is possible a smaller batch size needs to be used
conda activate comp

f1=/mounts/data/proj/antmarakis/multilingual/lms/tokenizers/wikipedia_en_$1_$3_$2/vocab.txt

if [ ! -e $f1 ] ; then
    python tokenization_wiki.py $1 $2 $3
fi

python concat_tokenizer.py wikipedia False False wikipedia_en_1G wiki_en_$1 wikipedia_$3 wiki_$3_$2

python run_language_modeling.py \
--model_type bert \
--config_name bert_base.json \
--train_data_file /mounts/data/proj/antmarakis/wikipedia/wikipedia_en_$3_seq.txt \
--output_dir /mounts/data/proj/antmarakis/wikipedia/lms/wikipedia_en_$1_$3_$2_2_seq_128 \
--do_train \
--mlm \
--seed 0 \
--block_size 128 \
--per_gpu_train_batch_size 40 \
--num_train_epochs 2 \
--overwrite_output_dir \
--tokenizer_name /mounts/data/proj/antmarakis/multilingual/lms/tokenizers/wikipedia_en_$1_$3_$2 \
--warmup_steps 1200 \
--learning_rate 1e-4 \
--weight_decay 0.01 \
--adam_epsilon 1e-6 \
--gradient_accumulation_steps 4 \
--save_steps 10000000 \
--line_by_line \


conda activate simple_trans
python xnli_eval.py $1 $2 $3 2 wikipedia_en_$1_$3_$2_2_seq_128 128
