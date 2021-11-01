import tokenizers
import sentencepiece as spm
from typing import Text, Dict, List
from utils.utils import get_logger
import os
LOG = get_logger(__name__)


class MyTokenizer(object):
    """docstring for Tokenizer"""

    def __init__(self, name: Text, file: Text = None):
        self.name = name
        if self.name == "BertWordPieceTokenizer":
            if file is None:
                # beware special handling of chinese characters
                tokenizer_class = tokenizers.BertWordPieceTokenizer(
                    vocab_file=file, strip_accents=False, lowercase=False, clean_text=False, handle_chinese_chars=False)
            else:
                tokenizer_class = tokenizers.Tokenizer.from_file(file)
        elif self.name == "CharBPETokenizer":
            if file is None:
                tokenizer_class = tokenizers.CharBPETokenizer(vocab_file=file)
            else:
                tokenizer_class = tokenizers.Tokenizer.from_file(file)
        elif self.name == "SentencePiece_unigram":
            if file is None:
                tokenizer_class = spm.SentencePieceTrainer
            else:
                # load model
                tokenizer_class = spm.SentencePieceProcessor(model_file=file)
        else:
            raise ValueError("Unknown Tokenizer.")
        self.model = tokenizer_class

    @property
    def from_tokenizers(self) -> bool:
        return self.name in ["BertWordPieceTokenizer", "CharBPETokenizer"]

    @property
    def from_sentencepiece(self) -> bool:
        return self.name in ["SentencePiece_unigram"]

    def get_vocabsize(self) -> int:
        if self.from_tokenizers:
            raise NotImplementedError
        elif self.from_sentencepiece:
            return len(self.model)

    def train(self, input_file: Text, vocab_size: int, output_dir: Text):
        if self.from_tokenizers:
            self.model.train([input_file], vocab_size=vocab_size, limit_alphabet=int(1e9), min_frequency=2)
            # And finally save it somewhere
            self.model.save(os.path.join(output_dir, "tokenizer.model"))
        elif self.from_sentencepiece:
            self.model.train(input=input_file,
                             model_prefix=os.path.join(output_dir, "tokenizer"),
                             vocab_size=vocab_size,
                             user_defined_symbols=[],
                             hard_vocab_limit=False,
                             max_sentencepiece_length=512,
                             split_by_unicode_script=False,
                             split_by_number=False,
                             split_by_whitespace=False)
            # load the model and overwrite the trainer
            self.model = spm.SentencePieceProcessor(model_file=os.path.join(output_dir, "tokenizer.model"))

    def tokenize(self, sentence: Text) -> List[Text]:
        if self.from_tokenizers:
            return self.model.encode(sentence).tokens
        elif self.from_sentencepiece:
            return self.model.encode(sentence, out_type=str)

    @classmethod
    def get_token(cls, word2index: Dict[Text, int], token: Text, strict: bool = True) -> int:
        if strict:
            if token in word2index:
                return word2index[token]
            else:
                return -1
        else:
            if token in word2index:
                return word2index[token]
            elif token.startswith("##") and token[2:] in word2index:
                return word2index[token[2:]]
            elif not token.startswith("##") and "##" + token in word2index:
                return word2index["##" + token]
            else:
                return -1
