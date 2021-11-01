from typing import Any, List, Text, Tuple, Dict
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import entropy
import numpy as np
from utils.utils import get_logger
LOG = get_logger(__name__)


class Structure(object):
    """docstring for Structure"""

    def __init__(self):
        pass

    @staticmethod
    def eigenvector_sub(E: Any, n: int) -> Tuple[np.ndarray, int]:
        # length normalize
        E.normalize()
        # get nns
        if n > -1:
            X = E.X[:n, :]
        else:
            X = E.X.copy()
        LOG.info("Getting nearest neighbours...")
        dist = cosine_distances(X)
        # black out diagonal
        np.fill_diagonal(dist, 99)
        nns = np.argsort(dist, axis=1)[:, :1]
        # create adjacecny matrix and degree matrix
        A = np.zeros((X.shape[0], X.shape[0]))
        for i, j in enumerate(nns.flatten()):
            A[i, j] = 1
            # assume undirected graph
            A[j, i] = 1
        D = np.diag(A.sum(axis=0))
        # compute laplacian
        L = D - A
        # eigendecomposition
        # L is symmetric
        LOG.info("Computing Eigendecomposition...")
        lamb, _ = np.linalg.eigh(L)
        lamb.sort()
        lamb = lamb[::-1]
        sigma = lamb.sum()
        # get smallest k
        acc = 0.0
        for i, l in enumerate(lamb):
            if acc / sigma < 0.9:
                acc += l
            else:
                break
        k = i + 1
        return lamb, k

    def eigenvector(self, E: Any, F: Any, n: int) -> float:
        # could be critisized? Only measure for isospectrality
        lambda_E, k_E = self.eigenvector_sub(E, n)
        lambda_F, k_F = self.eigenvector_sub(F, n)
        k = min(k_E, k_F)
        return np.square(lambda_E[:k] - lambda_F[:k]).sum()

    def hausdorff(self, E: Any, F: Any) -> float:
        import gudhi
        # take from https://github.com/joelmoniz/BLISS/blob/master/gh/gh.ipynb

        def compute_diagram(x, homo_dim=1):
            rips_tree = gudhi.RipsComplex(x).create_simplex_tree(max_dimension=homo_dim)
            rips_diag = rips_tree.persistence()
            return [rips_tree.persistence_intervals_in_dimension(w) for w in range(homo_dim)]

        def compute_distance(x, y, homo_dim=1):
            diag_x = compute_diagram(x, homo_dim=homo_dim)
            diag_y = compute_diagram(y, homo_dim=homo_dim)
            return min([gudhi.bottleneck_distance(x, y, e=0) for (x, y) in zip(diag_x, diag_y)])

        distX = cosine_distances(E.X)
        distY = cosine_distances(F.X)
        result = compute_distance(distX, distY)
        return result

    def spectral(self, E: Any, F: Any, use_first_n: int) -> float:
        # see https://www.aclweb.org/anthology/2020.emnlp-main.186.pdf
        if E.X.shape[0] < E.X.shape[1] or F.X.shape[0] < F.X.shape[1]:
            #import ipdb;ipdb.set_trace()
            print("WARNING: Cannot compare singular values due to different rank.")
        sigma_e = np.linalg.svd(E.X, compute_uv=False)[:use_first_n]
        sigma_f = np.linalg.svd(F.X, compute_uv=False)[:use_first_n]
        svg = (np.square(np.log(sigma_e) - np.log(sigma_f)))[:use_first_n].sum()

        # effective rank
        sigma_e_normed = sigma_e / sigma_e.sum()
        sigma_f_normed = sigma_f / sigma_f.sum()

        effective_rank_e = int(np.exp(entropy(sigma_e_normed)))
        effective_rank_f = int(np.exp(entropy(sigma_f_normed)))

        cond_e = sigma_e[0] / sigma_e[-1]
        cond_f = sigma_f[0] / sigma_f[-1]

        eff_cond_e = sigma_e[0] / sigma_e[effective_rank_e]
        eff_cond_f = sigma_f[0] / sigma_f[effective_rank_f]

        cond_hm = 2 * cond_e * cond_f / (cond_e + cond_f)
        econd_hm = 2 * eff_cond_e * eff_cond_f / (eff_cond_e + eff_cond_f)
        return svg, cond_hm, econd_hm

    def relational(self):
        # requires translation pairs
        pass


class LexiconInduction(object):
    """docstring for LexiconInduction"""
    def __init__(self):
        pass

    @staticmethod
    def get_nn(X: np.ndarray ,Y: np.ndarray, n: int = 10) -> np.ndarray:
        dist = cosine_distances(X, Y)
        nns1 = np.argsort(dist, axis=1)[:, :n]
        return nns1

    def evaluate(self, E: Any, F: Any, gt: Any) -> Tuple[Dict, Dict]:
        raise ValueError("BUGGY!")
        # TODO is wrong, use only unique source words and multiple groundtruths. Also 
        # just use the official eval script
        E.get_mappings()
        F.get_mappings()
        # keep only training instances where both words are in the spaces
        gt = [x for x in gt if x[0] in E.Wset and x[1] in F.Wset]
        totals = len(gt)
        LOG.info("Evaluating on {} word pairs.".format(totals))
        Erelwords = [E.word2index[x[0]] for x in gt]
        Frelwords = [F.word2index[x[1]] for x in gt]
        # get distances and nearestneighbours
        nns1 = self.get_nn(E.X[Erelwords], F.X)
        nns2 = self.get_nn(F.X[Frelwords], E.X)
        ks = [1, 5, 10]
        p = {"E2F": {k: 0 for k in ks}, "F2E": {k: 0 for k in ks}}
        for i, (wE, wF) in enumerate(gt):
            predE = [F.index2word[j] for j in nns1[i, :]]
            predF = [E.index2word[j] for j in nns2[i, :]]
            for k in ks:
                if wF in predE[:k]:
                    p["E2F"][k] += 1
                if wE in predF[:k]:
                    p["F2E"][k] += 1
        for k in ks:
            p["E2F"][k] = p["E2F"][k] / totals
            p["F2E"][k] = p["F2E"][k] / totals
        return p["E2F"], p["F2E"]

