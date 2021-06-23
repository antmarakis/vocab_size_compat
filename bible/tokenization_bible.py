import os, sys, shutil
from tokenizers import BertWordPieceTokenizer

en_voc, t_voc = int(sys.argv[1]), int(sys.argv[2])
lang = sys.argv[3]

tokenizer = BertWordPieceTokenizer(lowercase=False, strip_accents=False, clean_text=True)
tokenizer.train(files=['/mounts/data/proj/antmarakis/bible/eng-x-bible-newworld2013_corpus.train'], vocab_size=en_voc, special_tokens=[
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
])

shutil.rmtree('/mounts/data/proj/antmarakis/multilingual/lms/tokenizers/bible_en_{}'.format(en_voc), ignore_errors=True)
os.mkdir('/mounts/data/proj/antmarakis/multilingual/lms/tokenizers/bible_en_{}'.format(en_voc))
tokenizer.save_model('/mounts/data/proj/antmarakis/multilingual/lms/tokenizers/bible_en_{}'.format(en_voc))
print('/mounts/data/proj/antmarakis/multilingual/lms/tokenizers/bible_en_{}'.format(en_voc))

tokenizer = BertWordPieceTokenizer(lowercase=False, strip_accents=False, clean_text=True)
tokenizer.train(files=['/mounts/data/proj/antmarakis/bible/{}-x-bible-newworld_corpus.train'.format(lang)], vocab_size=t_voc, special_tokens=[
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
])

shutil.rmtree('/mounts/data/proj/antmarakis/multilingual/lms/tokenizers/bible_{}_{}'.format(lang, t_voc), ignore_errors=True)
os.mkdir('/mounts/data/proj/antmarakis/multilingual/lms/tokenizers/bible_{}_{}'.format(lang, t_voc))
tokenizer.save_model('/mounts/data/proj/antmarakis/multilingual/lms/tokenizers/bible_{}_{}'.format(lang, t_voc))
print('/mounts/data/proj/antmarakis/multilingual/lms/tokenizers/bible_{}_{}'.format(lang, t_voc))