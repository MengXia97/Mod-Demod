import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D

class MeanShift:
    ## Class Variables ##
    X = radius = bandwidth = max_itr = centroids = None

    ## Constructors ##
    def __init__(self, radius, bandwidth, max_itr=10):
        self.radius = radius
        self.bandwidth = bandwidth
        self.max_itr = max_itr
    #
    # ## Methods ##
    # def parse_data(self, file_path, sep):
    #     df = pd.read_csv(file_path, sep=sep, header=None)
    #     self.X = df.values

    def gen_data(self):
        self.X, _ = make_blobs(n_samples=100, centers=4, n_features=3, random_state=0)


    def clusterify(self):
        self._init_centroid()

        for i in range(self.max_itr):
            for j, centroid in enumerate(self.centroids):
                neighbours = self._neighbourhood_points(self.centroids[j])
                new_centroid = self._calc_new_centroid(self.centroids[j], neighbours)
                self.centroids[j] = new_centroid

    def _init_centroid(self):
        self.centroids = np.copy(self.X)

    def _neighbourhood_points(self, centroid):
        neighbours = []
        for i, old_centroid in enumerate(self.centroids):
            if self.euclidean_dist(old_centroid, centroid) <= self.radius:
                neighbours.append(old_centroid)

        return neighbours

    def _calc_new_centroid(self, centroid, neighbours):
        numer = denom = 0

        for neighbour in neighbours:
            weight = self._gaussian_kernel(self.euclidean_dist(neighbour, centroid))
            numer += (weight * neighbour)
            denom += weight

        return numer / denom

    def _gaussian_kernel(self, distance):
        return (1 / (self.bandwidth * np.sqrt(2 * np.pi))) \
               * np.exp(-0.5 * (distance / self.bandwidth) ** 2)

    def euclidean_dist(self, p1, p2):
        sqr_dist = 0

        for i in range(len(p1)):
            sqr_dist += (p1[i] - p2[i]) ** 2

        return np.sqrt(sqr_dist)


def main():
    mean_shift = MeanShift(radius=2, bandwidth=5)
    mean_shift.gen_data()
    mean_shift.clusterify()
    # print(mean_shift.X)
    print(mean_shift.centroids)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(mean_shift.X[:, 0], mean_shift.X[:, 1], mean_shift.X[:, 2])
    # plt.show()
    # #

    for i, centroids in enumerate(mean_shift.centroids):
        ax.scatter(mean_shift.centroids[i][0], mean_shift.centroids[i][1], mean_shift.centroids[i][2], color='r', marker='x')
        # plt.scatter(mean_shift.centroids[i][0], mean_shift.centroids[i][1], mean_shift.centroids[i][2], color='r', marker='x')
    plt.show()


    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(mean_shift.X[:, 0], mean_shift.X[:, 1], mean_shift.X[:, 2])
    # plt.show()
    # for i in range(len(mean_shift_result)):
    #     # plt.scatter(X[i, 0], X[i, 1], X[i, 2], color=color[mean_shift_result[i]])
    #     ax.scatter(mean_shift.X[i, 0], mean_shift.X[i, 1], mean_shift.X[i, 2])
    # plt.show()


if __name__ == "__main__":
    main()

    # temp = data - cluster_centroid * np.ones((len(data), 3))
    # print(temp.shape)
    # dist = np.sqrt(np.sum(np.power(temp, 2), axis=1))
    # print(dist)
    # v = np.where(dist <= radius)
    # print(v)
    # temp_data = data[v]
    # cluster_frequency[i] = temp_data.shape[0]