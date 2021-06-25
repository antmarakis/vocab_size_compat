import collections
from typing import Text, List, Callable, Dict, Union, Any, Tuple
import numpy as np
from tqdm import tqdm
import argparse
from embedder import EmbeddingTrainer, TokenizerWrapper
#from tokenizer import Tokenizer
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
import numpy as np
import json
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from data import PBC


class Measure(object):
    """docstring for Measure"""
    def __init__(self, data: Any, tokenizer: Any):
        super(Measure, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        # get char numbers
        self.char_tokens, self.char_types, self.char_types_effective = self.get_tokenization_length(0)

    def get_tokenization_length(self, vocab_size: int):
        self.tokenizer.fit(self.data, vocab_size)
        total = 0
        types = set()
        for sentence in self.data:
            tokenized_sentence = self.tokenizer.tokenize(sentence)
            total += len(tokenized_sentence)
            types = types | set(tokenized_sentence)
        return total, self.tokenizer.tokenizer.get_vocab_size(), len(types)

    def get_measures(self, vocab_size: int) -> Tuple:
        # get some basic numbers
        self.tokens, self.types, self.types_effective = self.get_tokenization_length(vocab_size)
        # cr
        cr = self.tokens / self.char_tokens
        nominal_relcr = (self.tokens / self.types) / (self.char_tokens / self.char_types)
        effective_relcr = (self.tokens / self.types_effective) / (self.char_tokens / self.char_types_effective)
        return cr, nominal_relcr, effective_relcr


class VocabSizeSampler(object):
    """docstring for VocabSizeSampler
    TODO add uniform prior such that we get a more heavy tail"""

    def __init__(self, eps=1e-2, max_vocab=1e9):
        self.params = {}
        self.eps = eps
        self.max_vocab = max_vocab

    def load_parameters(self, file: Text):
        with open(file, "r") as fin:
            self.params = json.load(fin)

    def save_parameters(self, file: Text):
        outpath = os.path.join(file, "params.json")
        if os.path.exists(outpath):
            print(f"Warning, file exists: {outpath}")
        with open(file, "w") as outpath:
            json.dump(self.params, outpath)

    def sample(self, n_samples: int) -> List[int]:
        # sample using the inverse transform sampling
        uniforms = np.random.uniform(low=self.params["a"], high=1.0, size=n_samples)
        return [round(self.f_inv(u)) for u in uniforms]

    def f(self, x):
        return (1 - self.params["a"]) * (np.array(x) / self.params["c"]) ** self.params['beta'] + self.params["a"]

    def f_inv(self, x):
        return ((x - self.params["a"]) / (1 - self.params["a"])) ** (1 / self.params["beta"]) * self.params["c"]

    def f_grad(self, x):
        # not required
        return (1 - self.params["a"]) * self.params["beta"] * (1 / self.params["c"]) * (np.array(x) / self.params["c"]) ** (self.params['beta'] - 1)

    def f_grad_inv(self, x):
        # not required
        return self.params["c"] * (self.params["c"] * x / (self.params["beta"] * (1 - self.params["a"]))) ** (1 / (self.params["beta"] - 1))

    def plot_fit(self):
        xs, ys = zip(*self.compression.items())
        plt.scatter(xs, ys)
        plt.scatter(xs, [self.f(x) for x in xs], color="r")
        plt.show()

    def fit_regression(self, vocab_sizes: List[float], compression_rates: List[float]) -> None:
        # compute parameters
        # functional form: y = f_beta(x) = (1 - a) (x / c)^beta + a
        # where a = horizontal asymptote, i.e., min(compression_rates) - eps
        # and c = min(vocab_sizes), i.e., the number of characters in the corpus
        y = np.array(compression_rates)
        a = min(y) - self.eps
        y = (y - a) / (1 - a)
        xs = vocab_sizes
        c = min(xs)
        x = np.array(vocab_sizes) / c
        lr = LinearRegression(fit_intercept=False)
        lr.fit(np.log(x.reshape(-1, 1)), np.log(y))
        self.params = {"beta": float(lr.coef_),
                       "a": a,
                       "c": c,
                       "lower": int(self.vocab_size_min),
                       "upper": int(self.vocab_size_max)}

    def set_corpus(self, data: List[Text]) -> None:
        self.data = data

    def set_tokenizer(self, tokenizer: TokenizerWrapper):
        self.tok = tokenizer
        # get true max and min vocab size with the tokenizer
        self.tok.fit(self.data, 0)
        self.vocab_size_min = self.tok.tokenizer.get_vocab_size()
        self.denominator = self.get_tokenization_length()
        self.tok.fit(self.data, self.max_vocab)
        self.vocab_size_max = self.tok.tokenizer.get_vocab_size()
        print(f"Min vocab size: {self.vocab_size_min}")
        print(f"Max vocab size: {self.vocab_size_max}")

    def get_tokenization_length(self):
        length = 0
        for sentence in self.data:
            length += len(self.tok.tokenizer.encode(sentence).tokens)
        return length

    def get_compression_rates(self, outpath: Text, n_samples: int = 10) -> None:
        # do uniform sampling
        vocab_sizes = [self.vocab_size_min] + [int(x) for x in np.random.uniform(
            low=self.vocab_size_min, high=self.vocab_size_max, size=n_samples)] + [self.vocab_size_max]
        compression = {}
        for vocab_size in tqdm(vocab_sizes, desc="samples"):
            self.tok.fit(self.data, vocab_size)
            rate = self.get_tokenization_length() / max(self.denominator, 1)
            compression[self.tok.tokenizer.get_vocab_size()] = rate
            if outpath:
                os.path.makedirs(os.path.join(outpath, vocab_size))
                self.tok.save(os.path.join(outpath, vocab_size))
                with open(os.path.join(outpath, vocab_size, "rates.json"), "w") as fout:
                    json.dump(compression, fout)
        self.compression = compression
        compression_sorted = sorted(compression.items(), key=lambda x: x[0])
        self.fit_regression(*zip(*compression_sorted))
        if outpath:
            self.save_parameters(outpath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pbc", default="/nfs/datc/pbc/", type=str, required=False, help="")
    parser.add_argument("--data", default="", type=str, required=False, help="")
    parser.add_argument("--editions", default="", type=str, required=False, help="")
    parser.add_argument("--outpath", default="", type=str, required=True, help="")
    parser.add_argument("--step", default=0.1, type=float, required=False, help="")
    parser.add_argument("--max_vocab", default=100000, type=int, required=False, help="")
    parser.add_argument("--take_max_n", default=1000000, type=int, required=False, help="")
    args = parser.parse_args()


    if args.data:
        count = 0
        with open(args.data, "r") as fin:
            list_of_data = []
            for line in fin:
                if line.strip():
                    # TODO assumes that data is whitespace tokenized
                    count += 1
                    list_of_data.append(line.strip())
                    if count > args.take_max_n:
                        break
        editions = [(args.editions, args.editions)]
    else:
        pbc = PBC(args.pbc)

        if args.editions == "all":
            editions = reader.get_all_edition_names()
        elif args.editions == "newworld":
            editions = reader.get_all_edition_names()
            editions = [x for x in editions if "newworld" in x]
        else:
            editions = args.editions.split(",")

        editions = [(x.split("_")[0], x) for x in args.editions.split(",")]
        pbc.load_editions([x[1] for x in editions])

    results = []
    for lang, edition in editions:
        if args.data:
            pass
        else:
            list_of_data = pbc.text[lang][edition].values()
        tokwrapper = TokenizerWrapper("Wordpiece", "bert")
        sampler = VocabSizeSampler(eps=0.001, max_vocab=args.max_vocab)
        sampler.set_corpus(list_of_data)
        sampler.set_tokenizer(tokwrapper)
        sampler.get_compression_rates(None, 11)
        cr = 1.0
        results.append((edition, "max", 1.0, sampler.params["lower"]))
        while cr > sampler.params["a"]:
            results.append((edition, "f", cr, sampler.f_inv(cr)))
            print("{} {} {:.2f} {:.0f}".format(*results[-1]))
            cr -= args.step
        results.append((edition, "min", sampler.params["a"], sampler.params["upper"]))

    with open(args.outpath, "a") as fout:
        for result in results:
            fout.write(("{} {} {:.2f} {:.0f}\n".format(*result)))


if __name__ == '__main__':
    main()
