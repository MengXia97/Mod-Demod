import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D
import cv2
from scipy import signal
import math
import skimage.draw

SIZE = 8
BIN_SIZE = 9
def difference_filter(img):
    Dx = np.array([[-1, 0, 1]])
    Dy = Dx.T
    Ix = signal.convolve2d(img, Dx, mode='same')
    Iy = signal.convolve2d(img, Dy, mode='same')
    mag = np.sqrt(np.power(Ix, 2) + np.power(Iy, 2))
    mag = (mag - np.min(mag)) / (np.max(mag) - np.min(mag)) * 255

    theta = np.arctan2(Iy, Ix)
    # change the range into -pi/2 -- pi/2
    idx1, idx2 = np.where(theta < -np.pi / 2), np.where(theta > np.pi / 2)
    theta[idx1] += np.pi
    theta[idx2] -= np.pi

    return mag, theta


img = cv2.imread('100_32.jpg')
X = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mag, theta = difference_filter(X)

# # part1
# fig, ax1 = plt.subplots(1, 1)
# ax1.imshow(mag, cmap='gray')
# ax1.set_title('gradient magnitudes')
# fig.savefig('gradient magnitudes.png')
#
# fig, ax2 = plt.subplots(1, 1)
# ax2.imshow(theta, cmap='hsv')
# ax2.set_title('gradient orientations')
# fig.savefig('gradient orientations.png')

# part2
# padded
row, col = X.shape[0], X.shape[1]
if X.shape[0] % SIZE != 0:
    row = math.ceil(X.shape[0] / SIZE) * SIZE
if X.shape[1] % SIZE != 0:
    col = math.ceil(X.shape[1] / SIZE) * SIZE
theta_padded = np.zeros((row, col))
theta_padded[0: X.shape[0], 0: X.shape[1]] = theta
# 10 is an impossible number
theta_padded[np.where(mag < 0.1 * 255.0)] = 10

hist = np.zeros((int(row / SIZE), int(col / SIZE), BIN_SIZE))
for h in range(BIN_SIZE):
    for i in range(hist.shape[0]):
        for j in range(hist.shape[1]):
            sub_matrix = theta_padded[i * SIZE:(i + 1) * SIZE, j * SIZE:(j + 1) * SIZE]
            # normalization
            count = len(sub_matrix[sub_matrix != 10])
            if count == 0:
                break
            else:
                base = 1 / count
            cond1 = sub_matrix >= (-np.pi / 2 + np.pi * h / BIN_SIZE)
            cond2 = sub_matrix < (-np.pi / 2 + np.pi * (h + 1) / BIN_SIZE)
            count_h = len(sub_matrix[cond1 & cond2])
            hist[i, j, h] = count_h * base


def hogvis(descriptor, bsize, norient):
    d_h = descriptor.shape[0]
    d_w = descriptor.shape[1]
    hog_image = np.zeros((d_h * bsize, d_w * bsize), dtype=float)

    # radius of a spatial bin
    radius = bsize // 2
    orient = np.arange(norient)

    # angle of bin mid-points 0..pi
    orient_angle = (np.pi * (orient + .5) / norient)

    # end points of a line at each orientation
    vr = -(radius - 0.5) * np.cos(orient_angle)
    vc = (radius - 0.5) * np.sin(orient_angle)

    for r in range(d_h):
        for c in range(d_w):
            for o, dr, dc in zip(orient, vr, vc):
                centre = tuple([r * bsize + radius, c * bsize + radius])
                rr, cc = skimage.draw.line(int(centre[0] - dc), int(centre[1] + dr),
                                           int(centre[0] + dc), int(centre[1] - dr))
                hog_image[rr, cc] += descriptor[r, c, o]

    return hog_image

fig, ax = plt.subplots(1, 1)
ax.imshow(hogvis(hist, bsize=SIZE, norient=BIN_SIZE))
fig.savefig('hog_test.png')