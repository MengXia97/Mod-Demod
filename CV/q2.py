import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D
import cv2
from scipy import signal
import math

FILE = 'bsds_253027.jpg'
img = cv2.imread(FILE)


def difference_filter(img):
    Ix = np.zeros(img.shape)
    Iy = np.zeros(img.shape)
    mag = np.zeros(img.shape)
    theta = np.zeros((img.shape[0], img.shape[1]))

    Dx = np.array([[1, 0, -1]])
    Dy = Dx.T
    for i in range(3):
        Ix[:, :, i] = signal.convolve2d(img[:, :, i], Dx, mode='same')
        Iy[:, :, i] = signal.convolve2d(img[:, :, i], Dy, mode='same')
        mag[:, :, i] = np.sqrt(np.power(Ix[:, :, i], 2) + np.power(Iy[:, :, i], 2))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            max_mag = np.max(mag[i, j, :])
            if max_mag == mag[i, j, 0]:
                theta[i, j] = math.atan(Iy[i, j, 0] / Ix[i, j, 0])
            elif max_mag == mag[i, j, 1]:
                theta[i, j] = math.atan(Iy[i, j, 1] / Ix[i, j, 1])
            else:
                theta[i, j] = math.atan(Iy[i, j, 2] / Ix[i, j, 2])

    for i in range(3):
        mag[:, :, i] = (mag[:, :, i] - np.min(mag)) / (np.max(mag) - np.min(mag)) * 255

    return mag, theta

def derivative_gaussian_filter(img, sigma):
    Ix = np.zeros(img.shape)
    Iy = np.zeros(img.shape)
    mag = np.zeros(img.shape)
    theta = np.zeros((img.shape[0], img.shape[1]))

    Dx = np.array([[1, 0, -1]])
    Dy = Dx.T
    S = 21

    Hg = gaussFilter(S, S, sigma)
    Hx = signal.convolve2d(Hg, Dx, mode='same')
    Hy = signal.convolve2d(Hg, Dy, mode='same')

    # todo
    # figure;
    # mesh(Hx);
    # title('G*Dx')
    # figure;
    # mesh(Hy);
    # title('G*Dy');

    for i in range(3):
        Ix[:, :, i] = signal.convolve2d(img[:, :, i], Hx, mode='same')
        Iy[:, :, i] = signal.convolve2d(img[:, :, i], Hy, mode='same')
        mag[:, :, i] = np.sqrt(np.power(Ix[:, :, i], 2) + np.power(Iy[:, :, i], 2))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            max_mag = np.max(mag[i, j, :])
            if max_mag == mag[i, j, 0]:
                theta[i, j] = math.atan(Iy[i, j, 0] / Ix[i, j, 0])
            elif max_mag == mag[i, j, 1]:
                theta[i, j] = math.atan(Iy[i, j, 1] / Ix[i, j, 1])
            else:
                theta[i, j] = math.atan(Iy[i, j, 2] / Ix[i, j, 2])

    for i in range(3):
        mag[:, :, i] = (mag[:, :, i] - np.min(mag)) / (np.max(mag) - np.min(mag)) * 255
    return mag, theta

def oriented_filter(img, sigma):
    I1 = np.zeros(img.shape)
    I2 = np.zeros(img.shape)
    I3 = np.zeros(img.shape)
    I4 = np.zeros(img.shape)
    mag = np.zeros(img.shape)
    theta = np.zeros((img.shape[0], img.shape[1]))

    S = 21

    D1 = np.array([[0,0,0],[-1,0,1],[0,0,0]])
    D2 = np.array([[0,-1,0],[0,0,0],[0,1,0]])
    D3 = np.array([[-1,0,0],[0,0,0],[0,0,1]])
    D4 = np.array([[0,0,-1],[0,0,0],[1,0,0]])
    Hg = gaussFilter(S, S, sigma)

    H1 = signal.convolve2d(Hg, D1, mode='same')
    H2 = signal.convolve2d(Hg, D2, mode='same')
    H3 = signal.convolve2d(Hg, D3, mode='same')
    H4 = signal.convolve2d(Hg, D4, mode='same')

    # figure;
    # subplot(2, 2, 1), mesh(H1);
    # title('G*D1')
    # subplot(2, 2, 2), mesh(H2);
    # title('G*D2');
    # subplot(2, 2, 3), mesh(H3);
    # title('G*D3')
    # subplot(2, 2, 4), mesh(H4);
    # title('G*D4');
    for i in range(3):
        I1[:, :, i] = signal.convolve2d(img[:, :, i], H1, mode='same')
        I2[:, :, i] = signal.convolve2d(img[:, :, i], H2, mode='same')
        I3[:, :, i] = signal.convolve2d(img[:, :, i], H3, mode='same')
        I4[:, :, i] = signal.convolve2d(img[:, :, i], H4, mode='same')
        magxy = np.sqrt(np.power(I1[:, :, i], 2) + np.power(I2[:, :, i], 2))
        magdig = np.sqrt(np.power(I3[:, :, i], 2) + np.power(I4[:, :, i], 2))
        mag[:, :, i] = np.maximum(magxy, magdig)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            max_mag = np.max(mag[i, j, :])
            if max_mag == mag[i, j, 0]:
                theta[i, j] = math.atan(I2[i, j, 0] / I1[i, j, 0])
            elif max_mag == mag[i, j, 1]:
                theta[i, j] = math.atan(I2[i, j, 1] / I1[i, j, 1])
            else:
                theta[i, j] = math.atan(I2[i, j, 2] / I1[i, j, 2])

    for i in range(3):
        mag[:, :, i] = (mag[:, :, i] - np.min(mag)) / (np.max(mag) - np.min(mag)) * 255
    return mag, theta


def gaussFilter(n1,n2,std):
    h = np.zeros((n2, n1))
    for i in range(n2):
        for j in range(n1):
            u = [j - n1 / 2, i - n2/ 2]
            u = np.array(u)
            h[i, j] = gauss(u, std)
    return h / np.sum(h)


def gauss(x, std):
    return np.exp(-x.dot(x) / (2 * std**2))/ (std**2 * 2 * np.pi)


sigma = 4
# part1
# mag, theta = difference_filter(img)
# part2
# mag, theta = derivative_gaussian_filter(img, sigma)
# part3
mag, theta = oriented_filter(img,sigma)

#
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('gradient magnitudes')
ax1.imshow(mag[:, :, 0], cmap='gray')
ax1.set_title('R')
ax2.imshow(mag[:, :, 1], cmap='gray')
ax2.set_title('G')
ax3.imshow(mag[:, :, 2], cmap='gray')
ax3.set_title('B')
plt.show()

# gradient orientation
fig, ax = plt.subplots(1,1)
ax.imshow(theta, cmap='gray')
fig.suptitle('Gradient Orientation')
plt.show()

# gradient magnitudes after adding threshold
threshold = [0.1, 0.2, 0.5, 0.8]
max_mag = 255

for i in range(len(threshold)):
    temp_mag = mag
    temp_mag[np.where(temp_mag < threshold[i] * max_mag)] = 0
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle('gradient magnitudes when threshold = ', threshold[i])
    ax1.imshow(mag[:, :, 0], cmap='gray')
    ax1.set_title('R')
    ax2.imshow(mag[:, :, 1], cmap='gray')
    ax2.set_title('G')
    ax3.imshow(mag[:, :, 2], cmap='gray')
    ax3.set_title('B')
    plt.show()

ax = [1, 2, 3, 4]
fig, (ax[0], ax[1], ax[2], ax[3]) = plt.subplots(2, 2)
max_theta = np.max(theta)

for i in range(len(threshold)):
    temp_theta = theta
    temp_theta[np.where(theta < threshold[i] * max_theta)] = 0
    ax[i].imshow(temp_theta, cmap='gray')
    ax[i].set_title('gradient orientations when threshold = ', threshold[i])
plt.show()


