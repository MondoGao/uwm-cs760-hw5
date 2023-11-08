import numpy as np
import matplotlib.pyplot as plt


def main():
    sigmas = [0.5, 1, 2, 4, 8]
    for sigma in sigmas:
        sample(sigma)
    plt.show()


def sample(sigma: float):
    means = [[-1, -1], [1, -1], [0, 1]]
    covs = [[[2, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 2]], [[1, 0], [0, 2]]]
    covs = np.array(covs) * sigma

    samples = [sample_one(i, means[i], covs[i]) for i in range(len(means))]

    dataset = np.concatenate(samples)
    plt.scatter(dataset[:, 0], dataset[:, 1], c=dataset[:, 2])


def sample_one(label: int, mean: list, cov: list, num: int = 100):
    points = np.random.multivariate_normal(mean, cov, num)
    with_label = np.hstack((points, np.full((num, 1), label)))
    print(with_label)
    return with_label


if __name__ == "__main__":
    main()
