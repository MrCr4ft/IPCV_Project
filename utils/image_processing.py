import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import scipy
import scipy.signal


def binarizeImage(image: np.ndarray, kde_bandwidth: int = 5) -> np.ndarray:
    X = np.arange(255)[:, None]
    kde = KernelDensity(kernel='gaussian', bandwidth=kde_bandwidth).fit(image.ravel()[:, None])
    log_dens = kde.score_samples(X)
    threshold = scipy.signal.argrelmin(log_dens)[0][0]
    binarized_image = np.zeros(image.shape, dtype = image.dtype)
    binarized_image[image < threshold] = 0
    binarized_image[image >= threshold] = 255
    return threshold, 255 - binarized_image


def plot_connected_components(n_labels: int, label_ids: np.ndarray, stats: np.ndarray, centroids: np.ndarray):
    label_hue = (179 * (label_ids / (n_labels - 1))).astype("uint8")
    blank_ch = 255 * np.ones_like(label_hue)

    labeled_img = cv.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0

    #  source: https://stackoverflow.com/questions/46441893/connected-component-labeling-in-python

    for cc in range(1, n_labels):
        labeled_img = cv.rectangle(labeled_img,
                                   (stats[cc][0], stats[cc][1]),
                                   (stats[cc][0] + stats[cc][2], stats[cc][1] + stats[cc][3]),
                                   (255, 255, 255), 1)
        labeled_image = cv.circle(labeled_img, (int(centroids[cc][0]), int(centroids[cc][1])), radius=2,
                                  color=(0, 0, 0), thickness=-1)

    plt.imshow(labeled_img)
    plt.show()
