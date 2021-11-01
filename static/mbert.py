import argparse
from tokenizers import BertWordPieceTokenizer
import random
from embedder import TokenizerWrapper
from scipy.stats import entropy
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert-base-multilingual-cased", type=str, required=False, help="")
    parser.add_argument("--langs", default="", type=str, required=True, help="")
    parser.add_argument("--wiki", default="/mounts/work/philipp/data/wiki/", type=str, required=False, help="")
    parser.add_argument("--firstn", default=100, type=int, required=False, help="")
    parser.add_argument("--pick", default=0.2, type=float, required=False, help="")
    parser.add_argument("--argument2", action="store_true", help="")
    args = parser.parse_args()
    
    # tokenizer = transformers.BertTokenizer.from_pretrained(args.model)
    results = []
    for lang in sorted(args.langs.split(",")):
        data = []
        count = 0
        with open("{0}/wiki_{1}/wiki{1}.xml".format(args.wiki, lang), "r") as fin:
            for i, line in enumerate(fin):
                line = line.strip()
                if line and not (line.startswith("<doc") and line.endswith(">")) and not line.startswith("</doc>"):
                    if random.random() < args.pick:
                        count += 1
                        data.append(line.strip())
                        if count > args.firstn:
                            break

        basetokenizer = TokenizerWrapper("Wordpiece", "bert")
        basetokenizer.fit(data, 0)
        baselength = 0
        used_chars = set()
        for sentence in data:
            tokenized_sentence = basetokenizer.tokenize(sentence)
            baselength += len(tokenized_sentence)
            used_chars = used_chars | set(tokenized_sentence)

        charlength = baselength
        chartypes = basetokenizer.tokenizer.get_vocab_size()
        
        newtokenizer = TokenizerWrapper("Wordpiece", "bert")
        newtokenizer.tokenizer = BertWordPieceTokenizer("bert-base-multilingual-cased-vocab.txt")
        length = 0
        used_vocab = set()
        vocab_counter = Counter()
        for sentence in data:
            tokenized_sentence = newtokenizer.tokenize(sentence)
            length += len(tokenized_sentence)
            used_vocab = used_vocab | set(tokenized_sentence)
            for token in tokenized_sentence:
                vocab_counter[token] += 1

        types = len(used_vocab | used_chars)

        total = sum([x[1] for x in vocab_counter.most_common()])
        distribution = [x[1] / total for x in vocab_counter.most_common()]
        myentropy = entropy(distribution)
        results.append("{} {} {} {} {} {:.4f} {:.4f}".format(lang, length, types, charlength, chartypes, length / charlength, (length / types) / (charlength / chartypes)))

    for result in results:
        print(result)




if __name__ == '__main__':
    main()
