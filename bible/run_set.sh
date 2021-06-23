#!/bin/bash
conda activate comp
# args:
# 1 - set id (1, 2, 3)
# 2 - seed

f1=en_sizes_$1.txt
lines=`cat $f1`
for l1 in $lines; do

    f2='el_sizes.txt'
    while read l2; do
    source run_bible.sh $l1 $l2 -1 $2 ell 100
    done < $f2

    f2='ru_sizes.txt'
    while read l2; do
    source run_bible.sh $l1 $l2 -1 $2 rus 100
    done < $f2

    f2='zh_sizes.txt'
    while read l2; do
    source run_bible.sh $l1 $l2 -1 $2 zho 100
    done < $f2

done