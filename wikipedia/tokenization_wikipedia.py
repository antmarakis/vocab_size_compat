import os, sys, shutil
from tokenizers import BertWordPieceTokenizer

en_voc, other_voc, lang = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]

tokenizer = BertWordPieceTokenizer(lowercase=False, strip_accents=False, clean_text=True)
tokenizer.train(files=['/mounts/data/proj/antmarakis/wikipedia/wikipedia_en_1G.train'], vocab_size=en_voc, special_tokens=[
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
])

shutil.rmtree('/mounts/data/proj/antmarakis/multilingual/lms/tokenizers/wiki_en_{}'.format(en_voc), ignore_errors=True)
os.mkdir('/mounts/data/proj/antmarakis/multilingual/lms/tokenizers/wiki_en_{}'.format(en_voc))
tokenizer.save_model('/mounts/data/proj/antmarakis/multilingual/lms/tokenizers/wiki_en_{}'.format(en_voc))
print('/mounts/data/proj/antmarakis/multilingual/lms/tokenizers/wiki_en_{}'.format(en_voc))

tokenizer = BertWordPieceTokenizer(lowercase=False, strip_accents=False, clean_text=True, handle_chinese_chars=False)
tokenizer.train(files=['/mounts/data/proj/antmarakis/wikipedia/wikipedia_{}.train'.format(lang)], vocab_size=other_voc, special_tokens=[
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
])

shutil.rmtree('/mounts/data/proj/antmarakis/multilingual/lms/tokenizers/wiki_{}_{}'.format(lang, other_voc), ignore_errors=True)
os.mkdir('/mounts/data/proj/antmarakis/multilingual/lms/tokenizers/wiki_{}_{}'.format(lang, other_voc))
tokenizer.save_model('/mounts/data/proj/antmarakis/multilingual/lms/tokenizers/wiki_{}_{}'.format(lang, other_voc))
print('/mounts/data/proj/antmarakis/multilingual/lms/tokenizers/wiki_{}_{}'.format(lang, other_voc))