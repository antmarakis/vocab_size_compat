from data import PBC
from embedder import TokenizerWrapper
from get_vocab_sizes import Measure
import matplotlib.pyplot as plt
import argparse

model2cleanmodel = {"fastalign": r"fast-align", "eflomal": "eflomal",
                    "bert": "mBERT[8](Argmax)", "fasttext": "fastText(Argmax+Dist)"}

colors = {"cr": '#080D80',
          "cr2": '#7A0205',
          "eng_hin": '#5E66FF'}

markers = {"cr": 's',
           "cr2": 'P',
           "eng_hin": 'x'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pbc", default="/nfs/datc/pbc/", type=str, required=False, help="")
    parser.add_argument("--editions", default="", type=str, required=True, help="")
    parser.add_argument("--sizes", default="", type=str, required=True, help="")
    parser.add_argument("--outfile", default="", type=str, required=True, help="")
    parser.add_argument("--argument2", action="store_true", help="")
    args = parser.parse_args()
    pbc = PBC(args.pbc)
    pbc.load_editions(args.editions.split(","))

    sizes = [int(x) for x in args.sizes.split(",")]

    tok = TokenizerWrapper("Wordpiece", "")
    for edition in args.editions.split(","):
        lang = edition.split("_")[0]
        data = list(pbc.text[lang][edition].values())
        mymeasure = Measure(data, tok)
        results = {}
        for size in sizes:
            results[size] = mymeasure.get_measures(size)
        results = sorted(results.items(), key=lambda x: x[0])

        x = [el[0] for el in results]
        y1 = [el[1][0] for el in results]
        y2 = [el[1][1] for el in results]
        # plot
        plt.rcParams.update({'font.size': 13})
        fig, ax = plt.subplots()

        ax.plot(x, y1, label="ACR", color=colors["cr"], marker=markers["cr"], linewidth=3)
        ax.plot(x, y2, label="RCR", color=colors["cr2"], marker=markers["cr2"], linewidth=3)
        ax.set_xscale('log')
        ax.legend(loc='upper right', fontsize='small')
        plt.ylabel('', size=13)
        plt.ylim(-0.05, 1.0)
        plt.xlabel('n (vocab. size)', size=13)
        plt.grid(linestyle='-', alpha=0.5)
        plt.title("{}".format(lang))
        fig.set_size_inches(8, 3)
        plt.subplots_adjust(left=0.08, bottom=0.2, right=0.99, top=0.9, wspace=None, hspace=None)
        plt.savefig(args.outfile + lang + ".pdf")


if __name__ == '__main__':
    main()
