import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class KMeans:
    k: int
    iterations: int = 100

    def train(self, X):
        self.X = X
        # select randomly
        # shape(k, 2)
        centers = X[np.random.choice(X.shape[0], self.k, replace=False)]

        for i in range(self.iterations):
            # See: https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
            # compute distances to each center
            distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
            # get the closest center for each node
            labels = np.argmin(distances, axis=1)

            # Update the centroids
            for j in range(self.k):
                centers[j] = np.mean(X[labels == j], axis=0)

        self.centers = centers
        self.labels = labels

        return centers, labels

    def plot(self, plt=plt):
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.labels)
        plt.scatter(self.centers[:, 0], self.centers[:, 1], c="r", marker="+")

    def loss(self):
        n, d = self.X.shape
        y_pred = self.labels
        loss = 0
        for i in range(n):
            for k in range(self.k):
                if y_pred[i] == k:
                    loss += np.linalg.norm(self.X[i] - self.labels[k]) ** 2
        return loss

    def accuracy(self, y_real):
      pass
