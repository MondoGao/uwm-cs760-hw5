import numpy as np
import matplotlib.pyplot as plt

from hw5.k_means import KMeans
from hw5.gmm import GMM


def main():
    sigmas = [0.5, 1, 2, 4, 8]

    # data fig
    fig, axs = plt.subplots(len(sigmas))
    # cluster result fig
    fig2, axs2 = plt.subplots(2, len(sigmas))
    # loss and accuracy fig
    fig3, axs3 = plt.subplots(2, 2)

    kmeans_losses = np.empty((len(sigmas)))
    kmeans_accuracies = np.empty((len(sigmas)))
    gmm_losses = np.empty((len(sigmas)))
    gmm_accuracies = np.empty((len(sigmas)))
    for idx, sigma in enumerate(sigmas):
        data = sample(sigma, axs[idx])
        loss, accuracy = train_kmeans(data[:, :2], data[:, 2], axs2[0, idx])
        kmeans_losses[idx] = loss
        kmeans_accuracies[idx] = accuracy

        loss2, accuracy2 = train_gmm(data[:, :2], data[:, 2], axs2[1, idx])
        gmm_losses[idx] = loss2
        gmm_accuracies[idx] = accuracy2

    kmean_loss_fig = axs3[0, 0]
    kmean_accuracy_fig = axs3[0, 1]
    kmean_loss_fig.set_title("KMeans Loss")
    kmean_loss_fig.plot(sigmas, kmeans_losses)
    kmean_accuracy_fig.set_title("KMeans Accuracy")
    kmean_accuracy_fig.plot(sigmas, kmeans_accuracies)

    gmm_loss_fig = axs3[1, 0]
    gmm_accuracy_fig = axs3[1, 1]
    gmm_loss_fig.set_title("GMM Loss")
    gmm_loss_fig.plot(sigmas, kmeans_losses)
    gmm_accuracy_fig.set_title("GMM Accuracy")
    gmm_accuracy_fig.plot(sigmas, kmeans_accuracies)

    plt.show()


def train_kmeans(X, y_real, plt=plt):
    km = KMeans(k=3)
    km.train(X)
    km.plot(plt)
    loss = km.loss()
    accuracy = km.accuracy(y_real)
    return loss, accuracy


def train_gmm(X, y_real, plt=plt):
    gmm = GMM(k=3)
    gmm.train(X)
    gmm.plot(plt)
    loss = gmm.loss()
    accuracy = gmm.accuracy(y_real)
    return loss, accuracy


def sample(sigma: float, fig=plt):
    """
    Returns: (n, 3) array. The last column is the label.
    """
    means = [[-1, -1], [1, -1], [0, 1]]
    covs = [[[2, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 2]], [[1, 0], [0, 2]]]
    covs = np.array(covs) * sigma

    samples = [sample_one(i, means[i], covs[i]) for i in range(len(means))]

    dataset = np.concatenate(samples)
    fig.scatter(dataset[:, 0], dataset[:, 1], c=dataset[:, 2])
    fig.set_title(f"sigma = {sigma}")
    return dataset


def sample_one(label: int, mean: list, cov: list, num: int = 100):
    points = np.random.multivariate_normal(mean, cov, num)
    with_label = np.hstack((points, np.full((num, 1), label)))
    return with_label


if __name__ == "__main__":
    main()
