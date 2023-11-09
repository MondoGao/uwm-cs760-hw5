import numpy as np
import matplotlib.pyplot as plt

from hw5.k_means import KMeans


def main():
    sigmas = [0.5, 1, 2, 4, 8]

    # data fig
    fig, axs = plt.subplots(len(sigmas))
    # cluster result fig
    fig2, axs2 = plt.subplots(len(sigmas))
    # loss and accuracy fig
    fig3, axs3 = plt.subplots(2, 2)
    if len(sigmas) == 1:
        axs = [axs]
        axs2 = [axs2]

    losses = np.empty((len(sigmas)))
    accuracies = np.empty((len(sigmas)))
    for idx, sigma in enumerate(sigmas):
        data = sample(sigma, axs[idx])
        loss, accuracy = train_kmeans(data[:, :2], data[:, 2], axs2[idx])
        losses[idx] = loss
        accuracies[idx] = accuracy

    kmean_loss = axs3[0, 0]
    kmean_accuracy = axs3[0, 1]
    kmean_loss.set_title("KMeans Loss")
    kmean_loss.plot(sigmas, losses)
    kmean_accuracy.set_title("KMeans Accuracy")
    kmean_accuracy.plot(sigmas, accuracies)

    plt.show()


def train_kmeans(X, y_real, plt=plt):
    km = KMeans(k=3)
    km.train(X)
    km.plot(plt)
    loss = km.loss()
    accuracy = km.accuracy(y_real)
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
