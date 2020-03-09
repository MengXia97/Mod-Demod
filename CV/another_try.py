import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D
import cv2
FEATURES = 5
RADIUS = 10
img = cv2.imread('sunset.png')

shape = img.shape[0]
X = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
X = X.reshape((-1, 3))
pos = np.zeros((X.shape[0], 2))
for i in range(X.shape[0]):
    pos[i, 0] = int(i / shape)
    pos[i, 1] = int(i % shape)
X = np.hstack((X, pos))
peaks = np.zeros((X.shape[0], FEATURES))


#
# X, _ = make_blobs(n_samples=200, centers=4, n_features=3, random_state=0)
# peaks = np.zeros((200, 3))
# print(X.shape)
def mean_shift(data, radius):
    clusters = []
    for i in range(len(data)):
        print(i)
        cluster_centroid = data[i]
        cluster_frequency = np.zeros(len(data))
        # print(i)
        # Search points in circle
        while True:
            temp_data = []
            # temp = data - cluster_centroid * np.ones((len(data), 3))
            # z
            # np.sqrt(np.sum(np.power(c, 2), axis=1))

            temp = data - cluster_centroid * np.ones((len(data), FEATURES))
            dist = np.sqrt(np.sum(np.power(temp, 2), axis=1))
            v = np.where(dist <= radius)
            temp_data = data[v]
            cluster_frequency[i] = temp_data.shape[0]

            # for j in range(len(data)):
            #     v = data[j]
            #     # Handle points in the circles
            #     if np.linalg.norm(v - cluster_centroid) <= radius:
            #         temp_data.append(v)
            #         cluster_frequency[i] += 1

            # Update centroid
            old_centroid = cluster_centroid
            new_centroid = np.average(temp_data, axis=0)
            cluster_centroid = new_centroid
            # Find the mode
            if np.linalg.norm(new_centroid - old_centroid) <= 0.01:
                # np.array_equal(new_centroid, old_centroid):
                break

        # Combined 'same' clusters
        has_same_cluster = False

        for cluster in clusters:
            if np.linalg.norm(cluster['centroid'] - cluster_centroid) <= radius:
                has_same_cluster = True
                cluster['frequency'] = cluster['frequency'] + cluster_frequency
                break

        if not has_same_cluster:
            clusters.append({
                'centroid': cluster_centroid,
                'frequency': cluster_frequency
            })

        peaks[i] = cluster_centroid

    # print('clusters (', len(clusters), '): ', clusters)
    labels = clustering(data, clusters)

    # show_clusters(clusters, radius)
    return peaks, labels


# Clustering data using frequency
def clustering(data, clusters):
    t = []
    labels = []
    for cluster in clusters:
        cluster['data'] = []
        t.append(cluster['frequency'])
    t = np.array(t)

    # Clustering
    for i in range(len(data)):
        column_frequency = t[:, i]
        cluster_index = np.where(column_frequency == np.max(column_frequency))[0][0]
        clusters[cluster_index]['data'].append(data[i])
        labels.append(cluster_index)
    return labels


# Plot clusters
def show_clusters(clusters, radius):
    colors = 10 * ['r', 'b', 'y', 'g', 'c', 'm', 'k', 'w']
    fig = plt.figure()
    # plt.xlim((-8, 8))
    # plt.ylim((-8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # print(len(clusters))
    for i in range(len(clusters)):
        cluster = clusters[i]
        data = np.array(cluster['data'])
        # print(cluster)
        # ax.scatter(data[:, 0], data[:, 1], data[:, 2], color=colors[i])
        plt.scatter(data[:, 0], data[:, 1], data[:, 2], color=colors[i])
        # centroid = cluster['centroid']
        # ax.scatter(centroid[0], centroid[1], centroid[2], color=colors[i], marker='x')
        # x, y = np.cos(theta) * radius + centroid[0], np.sin(theta) * radius + centroid[1]
        # plt.plot(x, y, linewidth=1, color=colors[i])
    plt.show()


peaks, labels = mean_shift(X, RADIUS)

# print(peaks.shape)
# print(X.shape[0])
matrix = np.reshape(peaks[:, 0:3], (-1, shape, 3))

# print(matrix)
fig, ax = plt.subplots(1,1)
matrix = cv2.cvtColor(np.uint8(matrix), cv2.COLOR_LUV2RGB)
ax.imshow(cv2.cvtColor(matrix, cv2.COLOR_BGR2RGB))
plt.show()