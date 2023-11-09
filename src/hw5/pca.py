import numpy as np
from typing import Literal
from dataclasses import dataclass
from matplotlib import pyplot as plt


@dataclass
class PCA:
    d: int = 1
    mode: Literal["buggy", "demeaned", "normalized"] = "buggy"

    def train(self, X_train: np.ndarray):
        X = X_train.copy()
        self.X_mean = np.mean(X, axis=0)
        if self.mode == "demeaned" or self.mode == "normalized":
            X -= self.X_mean
            self.X_mean = np.mean(X, axis=0)
        if self.mode == "normalized":
            X = X / np.std(X, axis=0)
            self.X_mean = np.mean(X, axis=0)

        self.X = X

        # See: https://kozodoi.me/blog/20230326/pca-from-scratch
        cov = np.cov(X.T)
        eig_vals, eig_vecs = np.linalg.eig(cov)

        idx = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]
        self.components = eig_vecs[: self.d]

        X_pca = np.dot(X, self.components.T) + self.X_mean
        if self.mode == "buggy":
            X_pca -= self.X_mean
        self.X_pca = X_pca

        return X_pca, eig_vals, eig_vecs

    def plot(self, plt=plt):
        plt.set_title(f"PCA ({self.mode})")
        pca = self.X_pca
        plt.scatter(self.X[:, 0], self.X[:, 1], color="blue", marker="o")
        plt.scatter(pca[:, 0], pca[:, 1], color="red", marker="x")

    def error(self):
        return np.sum(np.square(self.X - self.X_pca))
