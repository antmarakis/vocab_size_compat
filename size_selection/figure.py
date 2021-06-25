import argparse
import collections
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator


task2pretty = {"svg": "SVG", 
                "cond-hm": "Cond-HM", 
                "econd-hm": "eCond-HM"}


def plot_hm():

    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", default=None, type=str, required=True, help="")
    parser.add_argument("--outfile", default=None, type=str, required=True, help="")
    parser.add_argument("--exid", default=None, type=str, required=True, help="")
    parser.add_argument("--l1", default=None, type=str, required=True, help="")
    parser.add_argument("--l2", default=None, type=str, required=True, help="")
    parser.add_argument("--task", default="spectral", type=str, required=False, help="")
    parser.add_argument("--use_rate", action="store_true", help="")
    parser.add_argument("--d3", action="store_true", help="")
    args = parser.parse_args()

    results = collections.defaultdict(dict)
    with open(args.infile, "r") as fin:
        for line in fin:
            exid, l1, size1, rate1, l2, size2, rate2, task, figure = line.strip().split(" ")
            if exid == args.exid and task == args.task and l1 == args.l1 and l2 == args.l2:
                if args.use_rate:
                    results[float(rate1)][float(rate2)] = float(figure)
                else:
                    results[int(size1)][int(size2)] = float(figure)

    matrix = []
    x = sorted(results.keys())
    long_x = []
    long_y = []
    long_z = []
    for r1, first in sorted(results.items(), key=lambda x: x[0]):
        matrix.append([])
        for r2, figure in sorted(first.items(), key=lambda x: x[0]):
            y = sorted(first.keys())
            long_x.append(r1)
            long_y.append(r2)
            long_z.append(figure)
            matrix[-1].append(figure)

    matrix = np.array(matrix)
    #print(matrix)
    # plt.matshow(matrix, cmap='cool', interpolation='nearest')
    # ax = sns.heatmap(matrix)
    #ax.savefig(args.outfile)
        # Make data.

    if args.d3:
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        if args.use_rate:
            Xm, Ym = np.meshgrid(x, y)
        else:
            Xm, Ym = np.meshgrid(np.log(x), np.log(y))
        #X, Y = np.meshgrid(x, y)
        Z = matrix.transpose()

        # Plot the surface.
        #ax.yaxis._set_scale('log')
        #ax.xaxis._set_scale('log')

        surf = ax.plot_surface(Xm, Ym, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        if args.use_rate:
            ax.set_xlabel("rate[{}]".format(args.l1.split("_")[0]))
            ax.set_ylabel("rate[{}]".format(args.l2.split("_")[0]))
            ax.set_zlabel("Spectral Similarity")
        else:
            ax.set_xlabel("log(n)[{}]".format(args.l1.split("_")[0]))
            ax.set_ylabel("log(n)[{}]".format(args.l2.split("_")[0]))
            ax.set_zlabel("Spectral Similarity")
        # Customize the z axis.

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
    else:
        if not args.use_rate:
            long_x = list(np.log(long_x))
            long_y = list(np.log(long_y))
        import matplotlib.tri as tri
        fig, ax = plt.subplots()
        triang = tri.Triangulation(long_x, long_y)
        interpolator = tri.LinearTriInterpolator(triang, long_z)
        Xeven, Yeven = np.linspace(min(long_x), max(long_x), 100), np.linspace(min(long_y), max(long_y), 100)
        xeven, yeven = np.meshgrid(Xeven, Yeven)
        zeven = interpolator(xeven, yeven)

        # Note that scipy.interpolate provides means to interpolate data on a grid
        # as well. The following would be an alternative to the four lines above:
        #from scipy.interpolate import griddata
        #zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')
        if args.task in ["cond-hm", "econd-hm"]:
            zeven = -zeven
        ax.contour(xeven, yeven, zeven, levels=15, linewidths=0.5, colors='k')
        cntr1 = ax.contourf(xeven, yeven, zeven, levels=15, cmap="RdBu_r")
        # fig.colorbar(cntr1, ax=ax)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if args.use_rate:
            ax.set_xlabel("compr. rate[{}]".format(args.l1.split("_")[0]))
            ax.set_ylabel("compr. rate[{}]".format(args.l2.split("_")[0]))
            ax.set_title(task2pretty[args.task])
        else:
            ax.set_xlabel("log(n)[{}]".format(args.l1.split("_")[0]))
            ax.set_ylabel("log(n)[{}]".format(args.l2.split("_")[0]))
            ax.set_title(task2pretty[args.task])
        fig.set_size_inches(3, 3)
        plt.subplots_adjust(left=0.2, bottom=0.15, right=0.95, top=0.9, wspace=None, hspace=None)
    plt.savefig(args.outfile)

if __name__ == '__main__':
    plot_hm()
