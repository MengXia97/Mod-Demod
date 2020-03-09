import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import time
import cv2

threshold = 0.01
cluster_threshold = 1
radius = 1

# img = cv2.imread('sunset.png')
# X = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
# Z_feature3 = X.reshape((-1, 3))
# X = np.float32(Z_feature3)


def main():
    X, _ = make_blobs(n_samples=100, centers=4, n_features=3, random_state=0)

    shift_points, labels = MeanShift().findPeak(X, kernel_bandwidth=radius)
    color = ['r', 'b', 'y', 'g']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print(labels)
    for i in range(len(labels)):
        ax.scatter(X[i, 0], X[i, 1], X[i, 2], color=color[labels[i]])
    plt.show()


class MeanShift:
    def findPeak(self, points, kernel_bandwidth):
        shiftingPoints = np.array(points)
        shifting = [True] * points.shape[0]

        while True:
            max_dist = 0

            for i in range(0, len(shiftingPoints)):
                print(i)
                if not shifting[i]:
                    continue
                p_shift_init = shiftingPoints[i].copy()
                shiftingPoints[i] = self.shift(shiftingPoints[i], points, kernel_bandwidth)
                dist = self.distance(shiftingPoints[i], p_shift_init)
                max_dist = max(max_dist, dist)
                shifting[i] = dist > threshold

            if max_dist < threshold:
                break

        # for i in range(0, len(shiftingPoints) - 1):
        #     for j in range(0, len(shiftingPoints) - 1):
        #         if self.distance(points[i], points[j]) < radius / 2:
        #             shiftingPoints[i] = (shiftingPoints[i] + shiftingPoints[j]) / 2
        #             shiftingPoints[j] = (shiftingPoints[i] + shiftingPoints[j]) / 2

        cluster_ids, shiftingPoints = self._cluster_points(shiftingPoints, shiftingPoints.tolist())
        return shiftingPoints, cluster_ids

    def shift(self, point, points, kernel_bandwidth):
        x, y, z, scale = 0, 0, 0, 0
        for p in points:
            dist = self.distance(point, p)
            weight = self.gaussian_kernel(dist, kernel_bandwidth)
            x += p[0] * weight
            y += p[1] * weight
            z += p[2] * weight
            scale += weight
        x, y, z = x / scale, y / scale, z / scale
        return [x, y, z]

    def _cluster_points(self, shiftingPoints, points):
        cluster_ids = []
        cluster_idx = 0
        cluster_centers = []

        for i, point in enumerate(points):
            if len(cluster_ids) == 0:
                cluster_ids.append(cluster_idx)
                cluster_centers.append(point)
                cluster_idx += 1
            else:

                # temp = cluster_centers - point * np.ones((len(cluster_centers), 3))
                # dist = np.sqrt(np.sum(np.power(temp, 2), axis=1))
                # v = np.where(dist < cluster_threshold)
                # cluster_ids = list(v)
                # shiftingPoints[i] = shiftingPoints[v]


                #
                for center in cluster_centers:
                    dist = self.distance(point, center)
                    if dist < cluster_threshold:
                        idx = cluster_centers.index(center)
                        cluster_ids.append(idx)
                        # speed up
                        shiftingPoints[i] = shiftingPoints[idx]
                if len(cluster_ids) < i + 1:
                    cluster_ids.append(cluster_idx)
                    cluster_centers.append(point)
                    cluster_idx += 1

        return cluster_ids, shiftingPoints

    def distance(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def gaussian_kernel(self, distance, bandwidth):
        return (1 / (bandwidth * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (distance / bandwidth) ** 2)


if __name__ == '__main__':
    main()
