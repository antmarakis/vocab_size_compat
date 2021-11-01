import os
from typing import Text, Any, Union, List, Text
from tqdm import tqdm
import collections
from utils.utils import get_logger
from tokenizer import MyTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace, BertPreTokenizer
import numpy as np
LOG = get_logger(__name__)


class TokenizerWrapper(object):
    def __init__(self, mode: Text, pretokenizer: Text):
        self.mode = mode
        self.pretokenizer = pretokenizer

    def get_tokenizer(self, vocab_size: int):
        if self.mode == "BPE":
            self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            if self.pretokenizer == "whitespace":
                self.tokenizer.pre_tokenizer = Whitespace()
            elif self.pretokenizer == "bert":
                self.tokenizer.pre_tokenizer = BertPreTokenizer()
            self.trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]",
                                      "[PAD]", "[MASK]"], vocab_size=vocab_size)
        elif self.mode == "Wordpiece":
            self.tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
            if self.pretokenizer == "whitespace":
                self.tokenizer.pre_tokenizer = Whitespace()
            elif self.pretokenizer == "bert":
                self.tokenizer.pre_tokenizer = BertPreTokenizer()
            self.trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=[
                                            "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def fit(self, data: Union[Text, List[Text]], vocab_size: int):
        self.get_tokenizer(vocab_size)
        if isinstance(data, str):
            self.tokenizer.train([data], trainer=self.trainer)
        else:
            self.tokenizer.train_from_iterator(data, trainer=self.trainer)

    def tokenize(self, sentence: Text) -> List[Text]:
        return self.tokenizer.encode(sentence).tokens

    def save(self, outpath: Text):
        self.tokenizer.save(outpath)

    def load(self, path: Text):
        self.tokenizer.from_file(path)


class Embeddings(object):
    """Class to load, edit and store word embeddings.

    Attr:
        X: embedding matrix
        W: list of words
        Wset: set of words
    """

    def __init__(self):
        """Initalize the wrapper

        Args:
            log: a logger object
        """
        pass

    def load(self, path, load_first_n=None, header=True):
        """Load word embeddings in word2vec format from a txt file.

        Args:
            path: path to the embedding file
            load_first_n: int; how many lines to load
            header: bool; whether the embedding file contains a header line
        """
        self.path = path
        LOG.info("loading embeddings: {}".format(self.path))

        fin = open(self.path, 'r')

        if header:
            n, d = map(int, fin.readline().split())
        else:
            n, d = None, None

        data = {}
        count = 0
        for line in tqdm(fin):
            count += 1
            if load_first_n is not None and count > load_first_n:
                break
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = list(map(float, tokens[1:]))

        self.W = list(data.keys())
        self.Wset = set(self.W)
        self.X = np.vstack(tuple([data[x] for x in self.W]))

        LOG.info("loaded {} / {} vectors with dimension {}.".format(len(self.W), n, self.X.shape[1]))

    def normalize(self):
        """Normalize the embeddings with l2 norm
        """
        self.X = (self.X.transpose() / np.linalg.norm(self.X, axis=1)).transpose()

    def filter(self, relevant):
        """Filter the embeddings to contain only words from "relevant".

        Args:
            relevant: iterable of words which should be kept
        """
        relevant = set(relevant)
        choose = []
        for word in self.W:
            if word in relevant:
                choose.append(True)
            else:
                choose.append(False)
        self.W = list(np.array(self.W)[choose])
        self.Wset = set(self.W)
        self.X = self.X[choose]

        LOG.info("filtered for {} / {} words.".format(len(relevant), len(self.W)))

    def store(self, fname):
        """Store the embedding space

        Args:
            fname: path to the file
        """
        outfile = open(fname, "w")
        n, dim = self.X.shape
        outfile.write("{} {}\n".format(n, dim))
        for i in range(n):
            outfile.write(self.W[i])
            for k in range(dim):
                outfile.write(" {}".format(self.X[i, k]))
            outfile.write("\n")
        outfile.close()

    def set_prefix(self, prefix):
        self.W = [prefix + x for x in self.W]
        self.Wset = set([prefix + x for x in self.Wset])

    @staticmethod
    def replace_prefix(prefix, word):
        return word.replace(prefix, "", 1)

    def remove_prefix(self, prefix):
        self.W = [self.replace_prefix(prefix, x) for x in self.W]
        self.Wset = set([self.replace_prefix(prefix, x) for x in self.Wset])

    def get_mappings(self):
        self.index2word = {i: w for (i, w) in enumerate(self.W)}
        self.word2index = {w: i for (i, w) in enumerate(self.W)}


def convert2wordspacefilebased(embeddings_src: Text, embeddings_tgt: Text, tokenizer_src: Text) -> Embeddings:
    src_emb = Embeddings()
    src_emb.load(embeddings_src)
    # get target words
    with open(embeddings_tgt, "r") as fp:
        tgt_vocab = []
        for line in fp:
            if line.strip():
                word = line.strip().split()[0]
                tgt_vocab.append(word)
    tok = MyTokenizer("BertWordPieceTokenizer", tokenizer_src)
    X = []
    source_dict = {w: i for (i, w) in enumerate(src_emb.W)}
    for word in tgt_vocab:
        tokens = tok.encode(word).tokens
        token_indices = [source_dict[token] for token in tokens]
        if len(token_indices) == 1:
            Xtmp = src_emb[token_indices]
        else:
            Xtmp = src_emb[token_indices].mean(axis=0)
        X.append(Xtmp)
    result = Embeddings()
    result.W = tgt_vocab
    result.Wset = set(tgt_vocab)
    result.X = np.array(X)
    return result


def convert2wordspace(embeddings_src: Embeddings, embeddings_tgt: Embeddings, tok: Any) -> Embeddings:
    # get target words
    tgt_vocab = embeddings_tgt.W
    X = []
    source_dict = {w: i for (i, w) in enumerate(embeddings_src.W)}
    for word in tgt_vocab:
        tokens = tok.tokenize(word)
        token_indices = [MyTokenizer.get_token(source_dict, x) for x in tokens]
        if not min(token_indices) > -1:
            print("No token in source embedding space: {}".format(tokens))
        token_indices = [x if x != -1 else source_dict["</s>"] for x in token_indices]
        Xtmp = embeddings_src.X[token_indices].mean(axis=0)
        X.append(Xtmp)
    result = Embeddings()
    result.W = tgt_vocab
    result.Wset = set(tgt_vocab)
    result.X = np.array(X)
    return result


class EmbeddingTrainer(object):
    """docstring for Embedding"""

    def __init__(self, input_file: Text, output_dir: Text) -> None:
        self.input_file = input_file
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def write_corpus(self):
        LOG.info("Writing corpus...")
        with open(os.path.join(self.output_dir, "corpus.txt"), "w") as fout, open(self.input_file, "r") as fin:
            for line in tqdm(fin):
                if line.strip():
                    tokenized = self.tokenizer.tokenize(line.strip())
                    fout.write("{}\n".format(" ".join(tokenized)))

    def get_vocabulary(self, vocabulary_size: int, tokenizer: Text, pretokenizer: Text) -> None:
        self.tokenizer = TokenizerWrapper(tokenizer, pretokenizer)
        self.tokenizer.fit(self.input_file, vocabulary_size)
        self.tokenizer.save(os.path.join(self.output_dir, "tokenizer.model"))
        # self.tokenizer = Tokenizer(tokenizer)
        # self.tokenizer.train(self.input_file, vocabulary_size, self.output_dir)

    def train(self, dim: int, subwords: bool = False) -> None:
        if subwords:
            minn, maxn = 3, 6
        else:
            minn, maxn = 0, 0
        command_old = """
        nice -n 19 /mounts/Users/cisintern/philipp/Dokumente/fastText-0.9.1/fasttext skipgram \
        -input {} \
        -output {} \
        -dim {} \
        -minCount 2 \
        -lr 0.025 \
        -epoch 15 \
        -neg 15 \
        -thread 48 \
        -minn {} \
        -maxn {}
        """
        command = """
        nice -n 19 /mounts/Users/cisintern/philipp/Dokumente/fastText-0.9.1/fasttext skipgram \
        -input {} \
        -output {} \
        -dim {} \
        -minCount 5 \
        -epoch 25 \
        -neg 5 \
        -thread 48 \
        -minn {} \
        -maxn {}
        """.format(os.path.join(self.output_dir, "corpus.txt"),
                   os.path.join(self.output_dir, "embeddings"),
                   dim,
                   minn,
                   maxn)
        os.system(command)


class MUSEDict(object):
    """docstring for Dictionary"""

    def __init__(self, path: Text = "/mounts/work/philipp/data/muse_evaluation/crosslingual/dictionaries"):
        self.path = path
        self.dict = {"all": {}, "train": {}, "test": {}}

    @staticmethod
    def _load_file(path: Text):
        dictionary = collections.defaultdict(list)
        with open(path, "r") as fp:
            for line in fp:
                if line.strip():
                    w1, w2 = line.strip().split()
                    dictionary[w1].append(w2)
        return dictionary

    def load(self, l1, l2):
        files = {"all": "{}-{}.txt".format(l1, l2),
                 "train": "{}-{}.0-5000.txt".format(l1, l2),
                 "test": "{}-{}.5000-6500.txt".format(l1, l2)}
        for k, v in files.items():
            self.dict[k] = self._load_file(os.path.join(self.path, v))


class VecMap(object):
    """docstring for VecMap"""

    def __init__(self):
        # TODO allow for supervision
        self.path = "/mounts/work/philipp/bert_alignment/vecmap"

    def map(self, inpath: Text, pathX: Text, pathY: Text, outpath: Text, orthogonal: False, vocabulary_cutoff: 20000):
        # todo add orthogonal and vocabualry cutoff options
        os.makedirs(outpath, exist_ok=True)
        if orthogonal:
            orthogonal = "--orthogonal"
        else:
            orthogonal = ""
        command = """python {0}/map_embeddings.py --unsupervised \
                    {1}/{2}/embeddings.vec \
                    {1}/{3}/embeddings.vec \
                    {4}/embeddings_src.vec \
                    {4}/embeddings_tgt.vec \
                    --cuda \
                    -v \
                    --vocabulary_cutoff {5} \
                    {6}""".format(self.path, inpath, pathX, pathY, outpath, vocabulary_cutoff, orthogonal)
        os.system(command)


class SID(object):
    """docstring for SID"""

    def __init__(self):
        pass

    def write_corpus(self, input_corpora, output_dir):
        pass
