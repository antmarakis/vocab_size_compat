#!/bin/bash
conda activate comp
# args:
# 1 - en size
# 2 - other size
# 3 - 0: skip, 1: run eval, other: run as normal
# 4 - seed
# 5 - other lang code (ell, rus, zho)
# 6 - epochs


f1=res_both/res_sem_joint_dev_$1_$2_$5.txt
f2=res_sem_joint_dev_$1_$2_$5.txt

if [ -e $f1 ] || [ -e $f2 ] ; then
    if [ $3 -eq 0 ] ; then
        echo "skipping $1 $2"
        return 0
    elif [ $3 -eq 1 ] ; then
        python semantic_search_both.py $1 $2 $5-x-bible-newworld bible_en_$1_$5_$2_$4_256_50
        return 0
    fi
fi

cd /mounts/data/proj/antmarakis/bible
python common_lines.py eng-x-bible-newworld2013.train $5-x-bible-newworld.train
python remove_verse_id.py eng-x-bible-newworld2013_new.train
python remove_verse_id.py $5-x-bible-newworld_new.train
cd ~/work/compatibility

python tokenization_bible.py $1 $2 $5
python concat_tokenizer.py bible False False eng-x-bible-newworld2013_corpus bible_en_$1 $5-x-bible-newworld_corpus bible_$5_$2

python run_language_modeling.py \
--model_type bert \
--train_data_file /mounts/data/proj/antmarakis/bible/en_$5_corpus.train \
--output_dir /mounts/data/proj/antmarakis/bible/lms/bible_en_$1_$5_$2_$4_256_$6 \
--config_name bert_small.json \
--do_train \
--mlm \
--seed $4 \
--block_size 256 \
--per_gpu_train_batch_size 100 \
--num_train_epochs $6 \
--overwrite_output_dir \
--tokenizer_name /mounts/data/proj/antmarakis/multilingual/lms/tokenizers/bible_en_$1_$5_$2 \
--warmup_steps 1200 \
--line_by_line \
--learning_rate 2e-3 \
--weight_decay 0.01 \
--adam_epsilon 1e-6 \
--gradient_accumulation_steps 1

python semantic_search_both.py $1 $2 $5-x-bible-newworld bible_en_$1_$5_$2_$4_256_$6
