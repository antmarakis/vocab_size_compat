"""
python concat_data_tokenizer.py wikipedia True False wikipedia_en_1M wikipedia_en_350 wikipedia_zh_2M wikipedia_zh_350
python concat_data_tokenizer.py wikipedia False False wikipedia_en_60M wiki_en_350 wikipedia_zh wiki_zh_3500
python concat_data_tokenizer.py bible True False eng-x-bible-kingjames_corpus bible_en_350 zho-x-bible-contemp_corpus bible_zh_350
python concat_data_tokenizer.py bible False False eng-x-bible-kingjames_corpus bible_en_350 zho-x-bible-contemp_corpus bible_zh_350

Concatenates two tokenizers by merging their vocabularies together,
both for the bible and wikipedia data. It can also concatenate fake language vocabularies.
"""

import sys, codecs, string, os, shutil

dataset = sys.argv[1]
id, id1, id2 = '', '', ''

if sys.argv[2] == 'True':
    id = '_id'
    id1 = '_id'

if sys.argv[3] == 'True':
    id = '_id'
    id2 = '_id'

f_names = sys.argv[4:] # just give the dataset names, WITHOUT extensions

if id1:
    datasets = ['/mounts/Users/cisintern/antmarakis/data/{}/{}_id.train'.format(dataset, f_names[0])]
    tokenizers = ['/mounts/data/proj/antmarakis/multilingual/lms/tokenizers/{}_id/vocab.txt'.format(f_names[1])]
else:
    datasets = ['/mounts/Users/cisintern/antmarakis/data/{}/{}.train'.format(dataset, f_names[0])]
    tokenizers = ['/mounts/data/proj/antmarakis/multilingual/lms/tokenizers/{}/vocab.txt'.format(f_names[1])]

if id2:
    datasets += ['/mounts/Users/cisintern/antmarakis/data/{}/{}_id.train'.format(dataset, f_names[2])]
    tokenizers += ['/mounts/data/proj/antmarakis/multilingual/lms/tokenizers/{}_id/vocab.txt'.format(f_names[3])]
else:
    datasets += ['/mounts/Users/cisintern/antmarakis/data/{}/{}.train'.format(dataset, f_names[2])]
    tokenizers += ['/mounts/data/proj/antmarakis/multilingual/lms/tokenizers/{}/vocab.txt'.format(f_names[3])]

specifier = '_'.join(['_'.join(t.split('_')[-2:]) for t in f_names[1::2]]) + id
print(specifier)


### MERGE TOKENIZERS ###
f = codecs.open(tokenizers[0], encoding='utf8')
t_all = [l.strip() for l in f.readlines()]
f.close()

for t in tokenizers[1:]:
    f = codecs.open(t, encoding='utf8')
    l = [l.strip() for l in f.readlines()]
    f.close()

    for token in l:
        if '[' in token or token in string.punctuation or token.isnumeric():
            continue

        if token in t_all:
            continue

        t_all += [token]


shutil.rmtree('/mounts/data/proj/antmarakis/multilingual/lms/tokenizers/{}_{}'.format(dataset, specifier), ignore_errors=True)
os.mkdir('/mounts/data/proj/antmarakis/multilingual/lms/tokenizers/{}_{}'.format(dataset, specifier))
print('/mounts/data/proj/antmarakis/multilingual/lms/tokenizers/{}_{}'.format(dataset, specifier))
shutil.copyfile('/mounts/Users/cisintern/antmarakis/work/compatibility/bert-base-cased-config.json', '/mounts/data/proj/antmarakis/multilingual/lms/tokenizers/{}_{}/config.json'.format(dataset, specifier))
f = codecs.open('/mounts/data/proj/antmarakis/multilingual/lms/tokenizers/{}_{}/vocab.txt'.format(dataset, specifier), 'w', encoding='utf8')
f.write('\n'.join(t_all))
f.close()