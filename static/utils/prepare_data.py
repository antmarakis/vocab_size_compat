import argparse
import os
import stanza
from tqdm import tqdm
import sys



def get_bible_data():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="")
    parser.add_argument("--editions", default=None, type=str, required=True, help="")
    args = parser.parse_args()
    from data import PBC
    reader = PBC("/nfs/datc/pbc/")
    if args.editions == "all":
        editions = reader.get_all_edition_names()
    elif args.editions == "newworld":
        editions = reader.get_all_edition_names()
        editions = [x for x in editions if "newworld" in x]
    else:
        editions = args.editions.split(",")
    # editions = ["ell_newworld", "eng_newworld2013", "zho_newworld", "rus_newworld"]
    reader.load_editions(editions)
    for lang in reader.text:
        for edition in reader.text[lang]:
            with open(os.path.join(args.output_dir, edition + ".txt"), "w") as fout:
                for _, text in reader.text[lang][edition].items():
                    fout.write("{}\n".format(text))


def get_wiki_data():
    from nltk import sent_tokenize
    import random
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None, type=str, required=True, help="")
    parser.add_argument("--input", default=None, type=str, required=True, help="")
    parser.add_argument("--take_max_n", default=None, type=int, required=False, help="")
    parser.add_argument("--samplingrate", default=None, type=float, required=False, help="")
    parser.add_argument("--sentence_tokenize", action="store_true", help="")
    args = parser.parse_args()
    count = 0
    with open(args.input, "r") as fin:
        with open(args.output, "w") as fout:
            for line in tqdm(fin):
                line = line.strip()
                if line.startswith("<doc ") and line.endswith(">"):
                    continue
                if line == "</doc>":
                    continue
                if line:
                    if args.sentence_tokenize:
                        sentences = sent_tokenize(line)
                    else:
                        sentences = [line]
                    for sentence in sentences:
                        if args.take_max_n and count > args.take_max_n:
                            sys.exit()
                            continue
                        if args.take_max_n and random.random() > args.samplingrate:
                            continue
                        count += 1
                        fout.write("{}\n".format(sentence))




def preprocess():
    # DEPRECATED / NOT USED
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=None, type=str, required=True, help="")
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="")
    parser.add_argument("--langs", default=None, type=str, required=True, help="")
    args = parser.parse_args()
    args.langs = args.langs.split(",")
    output_files = {lang: open(os.path.join(args.output_dir, lang + ".txt"), "w") for lang in args.langs}
    for year in os.listdir(args.input_dir):
        # get all documents
        docs = list(os.listdir(os.path.join(args.input_dir, year)))
        docs = set([x[:-5] for x in docs])
        for doc in docs:
            for lang in args.langs:
                with open(os.path.join(args.input_dir, year, "{}{}.txt".format(doc, lang)), "r") as fp:
                    sentence = ""
                    for line in fp:
                        pass
    # close all output_files
    for file in output_files.values():
        file.close()


def word_tokenize():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infolder", default=None, type=str, required=True, help="")
    parser.add_argument("--lang", default=None, type=str, required=True, help="")
    args = parser.parse_args()

    print(args.lang)
    # sentence tokenization either...
    text = []
    #count = 0
    with open(os.path.join(args.infolder, "{}.txt".format(args.lang))) as fin:
        for line in fin:
            if line.strip():
                #count += 1
                #if count > 99:
                #    break
                text.append(line.strip())
    #text = " ".join(text)
    # remove double tokenization
    text = "\n\n".join(text)
    nlp = stanza.Pipeline(lang=args.lang, processors='tokenize', tokenize_no_ssplit=True)
    doc = nlp(text)
    with open(os.path.join(args.infolder, "{}-tokenized.txt".format(args.lang)), "w") as fout:
        for i, sentence in enumerate(doc.sentences):
            to_write = [token.text for token in sentence.tokens]
            fout.write("{}\n".format(" ".join(to_write)))


if __name__ == '__main__':
    get_bible_data()
