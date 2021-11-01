from embedder import *
from evaluator import *
from utils.utils import get_logger
from tokenizer import MyTokenizer
import argparse
import collections
import re
LOG = get_logger(__name__)


def do_vocabembeds_multiplelevels(args):
    for lang in args.langs:
        LOG.info("Embeddings for {}-{}".format(lang, "mult"))
        trainer = EmbeddingTrainer(os.path.join(args.corpus, lang + ".txt"),
                                   os.path.join(args.output_dir, "{}-{}".format(lang, "mult")))
        with open(os.path.join(args.output_dir, "{}-{}".format(lang, "mult"), "corpus.txt"), "w") as fout:
            for vocab_size in args.vocab_sizes[lang]:
                with open(os.path.join(args.output_dir, "{}-{}".format(lang, vocab_size), "corpus.txt"), "r") as fin:
                    for line in tqdm(fin):
                        fout.write(line)
        trainer.train(args.dim)


def do_vocabembeds(args):
    for lang in args.langs:
        for vocab_size in args.vocab_sizes[lang]:
            LOG.info("Embeddings for {}-{}".format(lang, vocab_size))
            trainer = EmbeddingTrainer(os.path.join(args.corpus, lang + ".txt"),
                                       os.path.join(args.output_dir, "{}-{}".format(lang, vocab_size)))
            trainer.get_vocabulary(vocab_size, args.tokenizer_name, args.pretokenizer)
            trainer.write_corpus()
            trainer.train(args.dim)
        '''
        # get word level spaces
        LOG.info("Embeddings for {}-{}".format(lang, "wordlevel"))
        trainer = EmbeddingTrainer(None,
                                   os.path.join(args.output_dir, "{}-{}".format(lang, "wordlevel")))
        # copy existing corpus
        os.system("cp {} {}".format(os.path.join(args.corpus, lang + "-tokenized.txt"),
                                    os.path.join(args.output_dir, "{}-{}".format(lang, "wordlevel"), "corpus.txt")))
        trainer.train(args.dim)
        '''


def do_mapping(args):
    for vocab_size_e in args.vocab_sizes[args.l1]:
        # todo we do not need to load them, right?
        E = Embeddings()
        E.load(os.path.join(args.output_dir, "{}-{}".format(args.l1, vocab_size_e), "embeddings.vec"))
        for vocab_size_f in args.vocab_sizes[args.l2]:
            LOG.info("Mapping {}-{},{}-{}".format(args.l1, vocab_size_e, args.l2, vocab_size_f))
            # todo we do not need to load them, right?
            F = Embeddings()
            F.load(os.path.join(args.output_dir, "{}-{}".format(args.l2, vocab_size_f), "embeddings.vec"))
            vm = VecMap()
            vm.map(args.output_dir, "{}-{}".format(args.l1, vocab_size_e), "{}-{}".format(args.l2, vocab_size_f),
                   os.path.join(args.output_dir, "{}-{},{}-{}".format(args.l1, vocab_size_e, args.l2, vocab_size_f)),
                   orthogonal=args.orthogonal, vocabulary_cutoff=args.vocabulary_cutoff)


def do_mappingonwordlevel(args):
    # load word level stuff
    Ewordlevel = Embeddings()
    Ewordlevel.load(os.path.join(args.output_dir, "{}-{}".format(args.l1, "wordlevel"), "embeddings.vec"))
    Fwordlevel = Embeddings()
    Fwordlevel.load(os.path.join(args.output_dir, "{}-{}".format(args.l2, "wordlevel"), "embeddings.vec"))
    for vocab_size_e in [1000]:
        for vocab_size_f in [1000]:
            E = Embeddings()
            E.load(os.path.join(args.output_dir, "{}-{}".format(args.l1, vocab_size_e), "embeddings.vec"))
            F = Embeddings()
            F.load(os.path.join(args.output_dir, "{}-{}".format(args.l2, vocab_size_f), "embeddings.vec"))
            Econverted = convert2wordspace(E, Ewordlevel, MyTokenizer.from_file(os.path.join(
                args.output_dir, "{}-{}".format(args.l1, vocab_size_e), "vocab.json")))
            Fconverted = convert2wordspace(F, Fwordlevel, MyTokenizer.from_file(os.path.join(
                args.output_dir, "{}-{}".format(args.l2, vocab_size_f), "vocab.json")))
            Econverted.store(os.path.join(args.output_dir, "{}-{}".format(args.l1,
                                                                          vocab_size_e), "embeddings_wordlevel.vec"))
            Fconverted.store(os.path.join(args.output_dir, "{}-{}".format(args.l2,
                                                                          vocab_size_f), "embeddings_wordlevel.vec"))
            vm = VecMap()
            raise ValueError("Not implemented yet.")
            vm.map(args.output_dir, "{}-{}".format(args.l1, vocab_size_e), "{}-{}".format(args.l2, vocab_size_f),
                   os.path.join(args.output_dir, "{}-{},{}-{}".format(args.l1, vocab_size_e, args.l2, vocab_size_f)))

def do_wordlevelmapping(args):
    vm = VecMap()
    vm.map(args.output_dir, "{}-{}".format(args.l1, "wordlevel"), "{}-{}".format(args.l2, "wordlevel"),
               os.path.join(args.output_dir, "{}-{},{}-{}".format(args.l1, "wordlevel", args.l2, "wordlevel")))


def do_isoeval(args):
    for vocab_size_e in args.vocab_sizes[args.l1]:
        E = Embeddings()
        E.load(os.path.join(args.output_dir, "{}-{}".format(args.l1, vocab_size_e), "embeddings.vec"))
        for vocab_size_f in args.vocab_sizes[args.l2]:
            F = Embeddings()
            F.load(os.path.join(args.output_dir, "{}-{}".format(args.l2, vocab_size_f), "embeddings.vec"))
            analyser = Structure()
            svg, cond_hm, econd_hm = analyser.spectral(E, F, args.spectral_use_first_n)
            if args.vocab_sizes2comprate:
                rate1 = args.vocab_sizes2comprate[args.l1][vocab_size_e]
                rate2 = args.vocab_sizes2comprate[args.l2][vocab_size_f]
            else:
                rate1 = -1.0
                rate2 = -1.0
            with open(args.result_file, "a") as fp:
                fp.write("{} {} {} {} {} {} {} {} {}\n".format(args.exid, args.l1, vocab_size_e, rate1, args.l2, vocab_size_f, rate2, "svg", svg))
                fp.write("{} {} {} {} {} {} {} {} {}\n".format(args.exid, args.l1, vocab_size_e, rate1, args.l2, vocab_size_f, rate2, "cond-hm", cond_hm))
                fp.write("{} {} {} {} {} {} {} {} {}\n".format(args.exid, args.l1, vocab_size_e, rate1, args.l2, vocab_size_f, rate2, "econd-hm", econd_hm))
            # print(analyser.eigenvector(E, F, -1))
            # print(analyser.hausdorff(E, F))

def do_blieval(args):
    # load word level stuff
    Ewordlevel = Embeddings()
    Ewordlevel.load(os.path.join(args.output_dir, "{}-{}".format(args.l1, "wordlevel"), "embeddings.vec"))
    Fwordlevel = Embeddings()
    Fwordlevel.load(os.path.join(args.output_dir, "{}-{}".format(args.l2, "wordlevel"), "embeddings.vec"))
    # load dict
    gt = MUSEDict()
    gt.load(args.l1, args.l2)
    li = LexiconInduction()
    for vocab_size_e in vocab_sizes[args.l1]:
        for vocab_size_f in vocab_sizes[args.l2]:
            mapped_path = os.path.join(args.output_dir, "{}-{},{}-{}".format(args.l1,
                                                                             vocab_size_e, args.l2, vocab_size_f))
            E = Embeddings()
            E.load(os.path.join(mapped_path, "embeddings_src.vec"))
            F = Embeddings()
            F.load(os.path.join(mapped_path, "embeddings_tgt.vec"))
            # convert spaces
            Econverted = convert2wordspace(E, Ewordlevel, MyTokenizer.from_file(os.path.join(
                args.output_dir, "{}-{}".format(args.l1, vocab_size_e), "vocab.json")))
            Fconverted = convert2wordspace(F, Fwordlevel, MyTokenizer.from_file(os.path.join(
                args.output_dir, "{}-{}".format(args.l2, vocab_size_f), "vocab.json")))
            # eval
            p, pinv = li.evaluate(Econverted, Fconverted, gt.dict["test"])
            for k in p:
                with open(args.result_file, "a") as fp:
                    fp.write("{} {} {} {} {} {} {}\n".format(args.exid, args.l1,
                                                             vocab_size_e, args.l2, vocab_size_f, "p@{}".format(k), p[k]))
            for k in pinv:
                with open(args.result_file, "a") as fp:
                    fp.write("{} {} {} {} {} {} {}\n".format(args.exid, args.l1,
                                                             vocab_size_e, args.l2, vocab_size_f, "pinv@{}".format(k), pinv[k]))


def do_wordlevelblieval(args):
    gt = MUSEDict()
    gt.load(args.l1, args.l2)
    li = LexiconInduction()
    E = Embeddings()
    E.load(os.path.join(args.output_dir, "en-wordlevel,zh-wordlevel/embeddings_src.vec"))
    F = Embeddings()
    F.load(os.path.join(args.output_dir, "en-wordlevel,zh-wordlevel/embeddings_tgt.vec"))
    p, pinv = li.evaluate(E, F, gt.dict["test"])
    for k in p:
        with open(args.result_file, "a") as fp:
            fp.write("{} {} {} {} {} {} {}\n".format(args.exid, args.l1,
                                                     "wordlevel", args.l2, "wordlevel", "p@{}".format(k), p[k]))
    for k in pinv:
        with open(args.result_file, "a") as fp:
            fp.write("{} {} {} {} {} {} {}\n".format(args.exid, args.l1,
                                                     "wordlevel", args.l2, "wordlevel", "pinv@{}".format(k), pinv[k]))


def do_vecmapeval(args):
    Ewordlevel = Embeddings()
    Ewordlevel.load(os.path.join(args.output_dir, "{}-{}".format(args.l1, "wordlevel"), "embeddings.vec"))
    Fwordlevel = Embeddings()
    Fwordlevel.load(os.path.join(args.output_dir, "{}-{}".format(args.l2, "wordlevel"), "embeddings.vec"))
    for vocab_size_e in args.vocab_sizes[args.l1]:
        for vocab_size_f in args.vocab_sizes[args.l2]:
            mapped_path = os.path.join(args.output_dir, "{}-{},{}-{}".format(args.l1,
                                                                             vocab_size_e, args.l2, vocab_size_f))
            E = Embeddings()
            E.load(os.path.join(mapped_path, "embeddings_src.vec"))
            F = Embeddings()
            F.load(os.path.join(mapped_path, "embeddings_tgt.vec"))
            # convert spaces
            Econverted = convert2wordspace(E, Ewordlevel, MyTokenizer(args.tokenizer_name, os.path.join(
                args.output_dir, "{}-{}".format(args.l1, vocab_size_e), "tokenizer.model")))
            Fconverted = convert2wordspace(F, Fwordlevel, MyTokenizer(args.tokenizer_name, os.path.join(
                args.output_dir, "{}-{}".format(args.l2, vocab_size_f), "tokenizer.model")))
            Econverted.store(os.path.join(mapped_path, "embeddings_src_wordlevel.vec"))
            Fconverted.store(os.path.join(mapped_path, "embeddings_tgt_wordlevel.vec"))
            command = """python /mounts/work/philipp/bert_alignment/vecmap/eval_translation.py \
                    {} \
                    {} \
                    -d /mounts/work/philipp/data/muse_evaluation/crosslingual/dictionaries/{}-{}.0-5000.txt""".format(os.path.join(mapped_path, "embeddings_src_wordlevel.vec"), os.path.join(mapped_path, "embeddings_tgt_wordlevel.vec"), args.l1, args.l2)
            result = os.popen(command).read()
            LOG.info(result)
            result = re.findall(r"\d+\.\d+", result)
            coverage, accuracy = float(result[0]), float(result[1])
            with open(args.result_file, "a") as fp:
                fp.write("{} {} {} {} {} {} {}\n".format(args.exid, args.l1,
                                                         vocab_size_e, args.l2, vocab_size_f, "vm-coverage", coverage))
                fp.write("{} {} {} {} {} {} {}\n".format(args.exid, args.l1,
                                                         vocab_size_e, args.l2, vocab_size_f, "vm-accuracy", accuracy))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="/mounts/work/philipp/isom/data/multiun", type=str, required=False, help="")
    parser.add_argument("--output_dir", default="/mounts/work/philipp/isom/embeddings",
                        type=str, required=False, help="")
    parser.add_argument("--result_file", default="/mounts/work/philipp/isom/results/tmp.txt",
                        type=str, required=False, help="")
    parser.add_argument("--l1", default=None, type=str, required=True, help="")
    parser.add_argument("--l2", default=None, type=str, required=True, help="")
    parser.add_argument("--exid", default=None, type=str, required=True, help="")
    parser.add_argument("--pretokenizer", default="bert", type=str, required=False, help="")
    parser.add_argument("--tokenizer_name", default="Wordpiece", type=str, required=False, help="")
    parser.add_argument("--sizes", default=None, type=str, required=False, help="")
    parser.add_argument("--dim", default=100, type=int, required=False, help="")
    parser.add_argument("--spectral_use_first_n", default=40, type=int, required=False, help="")
    parser.add_argument("--do_vocabembeds", action="store_true", help="")
    parser.add_argument("--do_vocabembeds_multiplelevels", action="store_true", help="")
    parser.add_argument("--do_mapping", action="store_true", help="")
    parser.add_argument("--do_mappingonwordlevel", action="store_true", help="")
    parser.add_argument("--do_isoeval", action="store_true", help="")
    parser.add_argument("--do_blieval", action="store_true", help="")
    parser.add_argument("--do_vecmapeval", action="store_true", help="")
    parser.add_argument("--do_wordlevelblieval", action="store_true", help="")
    parser.add_argument("--do_wordlevelmapping", action="store_true", help="")
    parser.add_argument("--do_all", action="store_true", help="")
    parser.add_argument("--orthogonal", action="store_true", help="")
    parser.add_argument("--embed_onlyl2", action="store_true", help="")
    parser.add_argument("--vocabulary_cutoff", default=20000, type=int, required=False, help="")
    args = parser.parse_args()

    if args.embed_onlyl2:
        args.langs = [args.l2]
    else:
        args.langs = [args.l1, args.l2]

    if args.sizes:
        args.vocab_sizes = collections.defaultdict(set)
        args.vocab_sizes2comprate = collections.defaultdict(dict)
        with open(args.sizes, "r") as fin:
            for line in fin:
                edition, _ , rate, size = line.strip().split()
                args.vocab_sizes[edition].add(int(size))
                args.vocab_sizes2comprate[edition][int(size)] = float(rate)

        for edition, sizes in args.vocab_sizes.items():
            args.vocab_sizes[edition] = list(sorted(sizes))
        args.vocab_sizes = dict(args.vocab_sizes)
    else:
        args.vocab_sizes2comprate = {}
        args.vocab_sizes = {
            "en": [144, 300, 400, 500, 1000, 10000, 20000, 47039],
            "fakeen": [144, 300, 400, 500, 1000, 10000, 20000, 47039],
            "zh": [3162, 5800, 5900, 6000, 6500, 8000, 10000, 75465],
            "ar": [174, 350, 400, 500, 1000, 10000, 20000, 101750],
            "fr": [144, 300, 400, 500, 1000, 10000, 20000, 45303]}

        args.vocab_sizes = {
            "ell_newworld": [143, 167, 201, 251, 330, 475, 642, 821, 1147, 1651, 2156, 2524, 2661, 51077],
            "eng_newworld2013": [129, 147, 172, 209, 267, 374, 653, 957, 1785, 2612, 3440, 4139, 4268, 18865],
            "zho_newworld": [6189, 6567, 7054, 7722, 8733, 10592, 17083, 33786],
            "rus_newworld": [137, 161, 195, 244, 323, 471, 663, 827, 1190, 1716, 2243, 2632, 2769, 59341],
            "fin_newworld": [121, 142, 171, 213, 278, 391, 628, 1372, 21161, 50000, 77767],
            "arb_newworld": [131, 158, 197, 254, 346, 511, 871, 2024, 17921, 50000, 108492],
            "jpn_newworld": [3818, 4534, 5519, 6944, 9155, 12954, 20657, 42215, 207307, 369171],
            "deu_newworld": [173, 198, 233, 282, 358, 490, 781, 1961, 8659, 42228]}

    if args.do_vocabembeds or args.do_all:
        do_vocabembeds(args)
    if args.do_vocabembeds_multiplelevels or args.do_all:
        do_vocabembeds_multiplelevels(args)
    if args.do_mapping or args.do_all:
        do_mapping(args)
    if args.do_mappingonwordlevel:
        do_mappingonwordlevel(args)
    if args.do_wordlevelmapping:
        do_wordlevelmapping(args)
    if args.do_isoeval or args.do_all:
        do_isoeval(args)
    if args.do_blieval or args.do_all:
        do_blieval(args)
    if args.do_wordlevelblieval:
        do_wordlevelblieval(args)
    if args.do_vecmapeval:
        do_vecmapeval(args)



if __name__ == '__main__':
    main()
