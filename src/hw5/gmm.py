from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics.cluster import completeness_score


@dataclass
class GMM:
    k: int
    iterations: int = 100

    def train(self, X):
        self.X = X
        k = self.k
        iterations = self.iterations
        n, d = X.shape

        means = np.random.rand(k, d)
        covs = np.array([np.eye(d)] * k)
        weights = np.ones(k) / k
        probs = np.zeros((n, k))

        for i in range(iterations):
            # E-step: compute the probability of each data point
            # See: https://towardsdatascience.com/gaussian-mixture-model-clearly-explained-115010f7d4cf
            for j in range(k):
                probs[:, j] = weights[j] * self.gaussian_pdf(means[j], covs[j])
            probs /= probs.sum(axis=1)[:, np.newaxis]

            # M-step: update the parameters
            for j in range(k):
                means[j] = np.average(X, axis=0, weights=probs[:, j])
                covs[j] = (
                    np.dot((probs[:, j] * (X - means[j]).T), (X - means[j]))
                    / probs[:, j].sum()
                )
                weights[j] = probs[:, j].sum() / n

        self.means = means
        self.covs = covs
        self.weights = weights
        self.probs = probs
        self.labels = np.argmax(probs, axis=1)

        return means, covs, weights

    def gaussian_pdf(self, mean, cov):
        var = multivariate_normal(mean=mean, cov=cov)
        return var.pdf(self.X)

    def plot(self, plt=plt):
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.labels)
        plt.scatter(self.means[:, 0], self.means[:, 1], marker="+", c="r")
    
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
        return completeness_score(y_real, self.labels)
    

    
