import numpy as np
from matplotlib import pyplot as plt
from typing import Literal

from hw5.pca import PCA


def main():
    X = np.loadtxt("data/data2D.csv", delimiter=",")
    modes = ["buggy", "demeaned", "normalized"]
    fig, axes = plt.subplots(1, 3, figsize=(8, 8))
    fig.tight_layout()

    for idx, mode in enumerate(modes):
        do_pca(X, mode, plt=axes[idx])

    plt.show()


def do_pca(X, mode: Literal["buggy", "demeaned", "normalized"] = "buggy", plt=plt):
    pca = PCA(mode=mode)
    pca.train(X)
    pca.plot(plt)


if __name__ == "__main__":
    main()
