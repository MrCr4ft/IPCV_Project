import typing

import cv2 as cv
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.cluster import DBSCAN
import scipy
import scipy.signal
from matplotlib import pyplot as plt

MIN_ROD_AREA = 1280
MAX_ROD_AREA = 6000


def binarizeImage(image: np.ndarray, kde_bandwidth: int = 5) -> np.ndarray:
    x = np.arange(255)[:, None]
    kde = KernelDensity(kernel='gaussian', bandwidth=kde_bandwidth).fit(image.ravel()[:, None])
    log_dens = kde.score_samples(x)
    threshold = scipy.signal.argrelmin(log_dens)[0][0]
    binarized_image = np.zeros(image.shape, dtype=image.dtype)
    binarized_image[image < threshold] = 0
    binarized_image[image >= threshold] = 255
    return threshold, 255 - binarized_image


def drawConnectedComponents(n_labels: int, label_ids: np.ndarray,
                            stats: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    #  source: https://stackoverflow.com/questions/46441893/connected-component-labeling-in-python
    label_hue = (179 * (label_ids / (n_labels - 1))).astype("uint8")
    blank_ch = 255 * np.ones_like(label_hue)

    labeled_img = cv.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0

    for cc in range(1, n_labels):
        labeled_img = cv.rectangle(labeled_img,
                                   (stats[cc][0], stats[cc][1]),
                                   (stats[cc][0] + stats[cc][2], stats[cc][1] + stats[cc][3]),
                                   (255, 255, 255), 1)
        labeled_img = cv.circle(labeled_img, (int(centroids[cc][0]), int(centroids[cc][1])), radius=2,
                                color=(0, 0, 0), thickness=-1)

    return labeled_img


def getCCMask(labels, label) -> np.ndarray:
    mask = np.zeros_like(labels, dtype=np.uint8)
    mask[labels == label] = 255
    return mask


def getMoment(mask: np.ndarray, order: typing.Tuple[int, int], barycenter: typing.Tuple[float, float] = (0, 0)) -> int:
    i, j = np.where(mask)
    return np.sum(((i - barycenter[0]) ** order[0]) * ((j - barycenter[1]) ** order[1]))


def getAngleAndAxes(mask: np.ndarray, blob_area: int) -> \
        typing.Tuple[float, np.ndarray, np.ndarray]:
    major_axis = np.zeros((3,), dtype=np.float32)
    minor_axis = np.zeros((3,), dtype=np.float32)

    barycenter = (getMoment(mask=mask, order=(1, 0)) / blob_area, getMoment(mask=mask, order=(0, 1)) / blob_area)
    m_0_2 = getMoment(mask=mask, order=(0, 2), barycenter=barycenter)
    m_2_0 = getMoment(mask=mask, order=(2, 0), barycenter=barycenter)
    m_1_1 = getMoment(mask=mask, order=(1, 1), barycenter=barycenter)

    theta = -0.5 * np.arctan(2 * m_1_1 / (m_0_2 - m_2_0)) + np.pi / 2
    second_derivative = 2 * (m_0_2 - m_2_0) * np.cos(2 * theta) - 4 * m_1_1 * np.sin(2 * theta)
    if second_derivative < 0:
        theta += np.pi / 2

    alpha = -np.sin(theta)
    beta = np.cos(theta)

    major_axis[0] = alpha
    major_axis[1] = -beta
    major_axis[2] = beta * barycenter[0] - alpha * barycenter[1]

    minor_axis[0] = beta
    minor_axis[1] = alpha
    minor_axis[2] = -beta * barycenter[1] - alpha * barycenter[0]

    return theta, major_axis, minor_axis


def findMER(major_axis: np.ndarray, minor_axis: np.ndarray, edges: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
    d_ma_min, d_mi_min = np.inf, np.inf
    d_ma_max, d_mi_max = -np.inf, -np.inf
    norm_ma = np.sqrt(major_axis[0] * major_axis[0] + major_axis[1] * major_axis[1])
    norm_mi = np.sqrt(minor_axis[0] * minor_axis[0] + minor_axis[1] * minor_axis[1])
    c_points = np.zeros((4, 2), dtype=np.int32)
    v_points = np.zeros((4, 2), dtype=np.int32)

    for i in range(edges.shape[0]):
        d_ma = (major_axis[0] * edges[i][0][0] + major_axis[1] * edges[i][0][1] + major_axis[2]) / norm_ma
        d_mi = (minor_axis[0] * edges[i][0][0] + minor_axis[1] * edges[i][0][1] + minor_axis[2]) / norm_mi
        if d_ma < d_ma_min:
            d_ma_min = d_ma
            c_points[0, 0] = edges[i][0][0]
            c_points[0, 1] = edges[i][0][1]
        elif d_ma > d_ma_max:
            d_ma_max = d_ma
            c_points[1, 0] = edges[i][0][0]
            c_points[1, 1] = edges[i][0][1]
        if d_mi < d_mi_min:
            d_mi_min = d_mi
            c_points[2, 0] = edges[i][0][0]
            c_points[2, 1] = edges[i][0][1]
        elif d_mi > d_mi_max:
            d_mi_max = d_mi
            c_points[3, 0] = edges[i][0][0]
            c_points[3, 1] = edges[i][0][1]

    c_l = -np.matmul(c_points[:2, :], major_axis[:2])
    c_w = -np.matmul(c_points[2:, :], minor_axis[:2])
    v_points[:, 0] = np.matmul(np.flip(np.array(np.meshgrid(c_l, c_w)).T.reshape(-1, 2), 1),
                               [major_axis[1], -minor_axis[1]]) / np.cross(major_axis[:2], minor_axis[:2])
    v_points[:, 1] = np.matmul(np.array(np.meshgrid(c_l, c_w)).T.reshape(-1, 2),
                               [minor_axis[0], -major_axis[0]]) / np.cross(major_axis[:2], minor_axis[:2])

    return v_points, c_points


def getWidthAtBarycenter(edges: np.ndarray, major_axis: np.ndarray, minor_axis: np.ndarray) -> \
        typing.Tuple[float, np.ndarray, np.ndarray]:
    points_placement_wrt_major_axis = edges[:, :, 0] * major_axis[0] + edges[:, :, 1] * major_axis[1] + major_axis[2]
    contour_points_distance_from_minor_axis = np.abs(
        edges[:, :, 0] * minor_axis[0] + edges[:, :, 1] * minor_axis[1] + minor_axis[2]) / np.sqrt(
        minor_axis[0] ** 2 + minor_axis[1] ** 2)
    left_indexes = np.where(points_placement_wrt_major_axis < 0)[0]
    right_indexes = np.where(points_placement_wrt_major_axis >= 0)[0]
    left_extreme = edges[left_indexes[np.argmin(contour_points_distance_from_minor_axis[left_indexes])]]
    right_extreme = edges[right_indexes[np.argmin(contour_points_distance_from_minor_axis[right_indexes])]]

    return np.linalg.norm(left_extreme - right_extreme), left_extreme, right_extreme


def getHoleCenterAndDiameter(hole_contour: np.ndarray) -> typing.Tuple[np.ndarray, float]:
    center, _, _ = cv.fitEllipse(hole_contour)
    area = cv.contourArea(hole_contour)
    radius = np.sqrt(area / np.pi)

    return center, 2 * radius


def computeHaralickCircularity(edges: np.array, barycenter: typing.Tuple[float, float]) -> float:
    distances = np.sqrt(np.sum((edges - np.flip(np.array(barycenter))) ** 2, axis=2))
    mu_r = np.mean(distances)
    sigma_r = np.sqrt(np.mean((distances - mu_r) ** 2))
    return mu_r / sigma_r


def getRodTypeAndDescription(rod_mask: np.ndarray) -> typing.Dict:
    description = dict()
    contours, hierarchy = cv.findContours(rod_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    description["area"] = getMoment(rod_mask, (0, 0))
    description["barycenter"] = getMoment(rod_mask, (1, 0)) / description["area"], \
                                getMoment(rod_mask, (0, 1)) / description["area"]
    theta, major_axis, minor_axis = getAngleAndAxes(rod_mask, description["area"])
    v_points, c_points = findMER(major_axis, minor_axis, contours[0])

    description["length"] = np.linalg.norm(v_points[0, :] - v_points[1, :])
    description["width"] = np.linalg.norm(v_points[0, :] - v_points[2, :])
    description["angle"] = theta
    description["major_axis"] = major_axis
    description["minor_axis"] = minor_axis
    description["v_points"] = v_points
    description["c_points"] = c_points
    description["barycenter_width"], extreme_a, extreme_b = getWidthAtBarycenter(contours[0], major_axis, minor_axis)
    description["haralick_circularity"] = computeHaralickCircularity(contours[0], description["barycenter"])

    if len(contours) == 2:
        description["rod_type"] = "A"
        description["1st_hole_center"], description["1st_hole_diameter"] = getHoleCenterAndDiameter(contours[1])
    else:
        description["rod_type"] = "B"
        description["1st_hole_center"], description["1st_hole_diameter"] = getHoleCenterAndDiameter(contours[1])
        description["2nd_hole_center"], description["2nd_hole_diameter"] = getHoleCenterAndDiameter(contours[2])

    return description


def separateTouchingRods(gray_image: np.ndarray, binarized_image: np.ndarray,
                         connected_components: typing.Tuple[int, np.ndarray, np.ndarray, np.ndarray],
                         max_rod_area: int):
    n_labels, label_ids, stats, centroids = connected_components

    for cc_idx in range(1, n_labels):
        if stats[cc_idx][-1] > max_rod_area:
            contours, _ = cv.findContours(getCCMask(label_ids, cc_idx), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contour = contours[0]

            cvx_hull = cv.convexHull(contour, returnPoints=False)
            cvx_defects = cv.convexityDefects(contour, cvx_hull)
            cvx_defects_points = []
            for i in range(cvx_defects.shape[0]):
                _, _, index, _ = cvx_defects[i, 0]
                cvx_defects_points.append(tuple(contour[index][0]))

            cvx_defects_points = np.array(cvx_defects_points)

            responses = cv.cornerHarris(gray_image.astype("float32"), 2, 3, 0.04)
            responses = cv.dilate(responses, None)
            y, x = np.where(responses > 0.05 * responses.max())
            harris_corner_points = np.zeros((x.shape[0], 2))
            harris_corner_points[:, 0] = x
            harris_corner_points[:, 1] = y

            clustering = DBSCAN(eps=20, min_samples=2).fit(harris_corner_points)
            for cluster_label in np.unique(clustering.labels_):
                cluster_points = harris_corner_points[clustering.labels_ == cluster_label]
                center = np.mean(cluster_points, axis=0)
                cvx_defects_distances_from_cluster_center = np.linalg.norm(np.array(cvx_defects_points) - center,
                                                                           axis=1)
                p1_i, p2_i = tuple(np.argsort(cvx_defects_distances_from_cluster_center)[:2])
                p1 = cvx_defects_points[p1_i]
                p2 = cvx_defects_points[p2_i]
                cv.line(binarized_image, p1, p2, 0)


def getPointsForAxes(axis_angle: float, center: typing.Tuple[int, int], length: float):
    length = int(length * 0.75)

    alpha = -np.sin(axis_angle)
    beta = np.cos(axis_angle)

    p1 = (int(center[0] + length * beta),
          int(center[1] + length * alpha))

    p2 = (int(center[0] - length * beta),
          int(center[1] - length * alpha))

    return p1, p2


def drawMERSWithAxes(connected_components: typing.Tuple[int, np.ndarray, np.ndarray, np.ndarray],
                     rod_descriptions: typing.List[typing.Dict]) -> np.ndarray:
    n_labels, label_ids, stats, centroids = connected_components
    label_hue = (179 * (label_ids / (n_labels - 1))).astype("uint8")
    blank_ch = 255 * np.ones_like(label_hue)

    labeled_img = cv.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0

    for idx, rod_description in enumerate(rod_descriptions):
        barycenter = rod_description['barycenter']
        barycenter = (int(barycenter[1]), int(barycenter[0]))

        p_1_major_axis, p_2_major_axis = getPointsForAxes(rod_description["angle"], barycenter,
                                                          rod_description["length"])
        p_1_minor_axis, p_2_minor_axis = getPointsForAxes(rod_description["angle"] + np.pi / 2, barycenter,
                                                          rod_description["width"])

        cv.line(labeled_img, p_1_major_axis, p_2_major_axis, color=(255, 255, 255), thickness=1)
        cv.line(labeled_img, p_1_minor_axis, p_2_minor_axis, color=(255, 255, 255), thickness=1)

        v_points = rod_description['v_points']
        v1 = tuple(v_points[0, :])
        v2 = tuple(v_points[1, :])
        v3 = tuple(v_points[2, :])
        v4 = tuple(v_points[3, :])

        cv.line(labeled_img, v1, v3, (255, 255, 255), 1, cv.LINE_AA)
        cv.line(labeled_img, v2, v1, (255, 255, 255), 1, cv.LINE_AA)
        cv.line(labeled_img, v3, v4, (255, 255, 255), 1, cv.LINE_AA)
        cv.line(labeled_img, v4, v2, (255, 255, 255), 1, cv.LINE_AA)

        cv.circle(labeled_img, barycenter, radius=2, color=(0, 0, 0), thickness=-1)

    return labeled_img


def discardDistractorsAndArtifacts(connected_components: typing.Tuple[int, np.ndarray, np.ndarray, np.ndarray]):
    n_labels, label_ids, stats, centroids = connected_components
    valid_indexes = np.where(stats[:, -1] >= MIN_ROD_AREA)[0]
    n_labels = valid_indexes.shape[0]
    label_ids[np.where(np.logical_not(np.isin(label_ids, valid_indexes)))] = 0
    remaining_labels = np.unique(label_ids.ravel())
    new_labels = np.argsort(remaining_labels)
    mapping = {remaining_labels[i]: new_labels[i] for i in range(remaining_labels.shape[0])}
    label_ids = np.vectorize(mapping.get)(label_ids)

    return n_labels, label_ids, stats[valid_indexes], centroids[valid_indexes]


def printRodDescriptions(rods_descriptions: typing.List[typing.Dict]):
    for idx, rod_description in enumerate(rods_descriptions):
        print("The rod labeled %d is of type %s" % (idx, rod_description["rod_type"]))
        print("Its barycenter is at position %s, it has an orientation angle of %.2f degrees, "
              "its length is %.2f, its width is %.2f, and its width at the barycenter is %.2f"
              % (str(rod_description["barycenter"]), np.rad2deg(rod_description["angle"]),
                 rod_description["length"], rod_description["width"], rod_description["barycenter_width"]))
        if rod_description["rod_type"] == "A":
            print("The only hole in the rod is at position %s and has diameter equal to %.2f" % (
            str(rod_description["1st_hole_center"]), rod_description["1st_hole_diameter"]))
        else:
            print("The first hole in the rod is at position %s and has diameter equal to %.2f" %
                  (str(rod_description["1st_hole_center"]), rod_description["1st_hole_diameter"]))
            print("The second hole in the rod is at position %s and has diameter equal to %.2f" %
                  (str(rod_description["2nd_hole_center"]), rod_description["2nd_hole_diameter"]))
    print("\n\n")


def pipeline(gray_img: np.ndarray, smooth_img: bool):
    img = gray_img.copy()
    if smooth_img:
        img = cv.bilateralFilter(img, d=17, sigmaColor=24, sigmaSpace=12)

    threshold, binarized_img = binarizeImage(img, kde_bandwidth=6)
    preliminary_connected_components = cv.connectedComponentsWithStats(binarized_img, connectivity=4)
    preliminary_connected_components = discardDistractorsAndArtifacts(preliminary_connected_components)

    separateTouchingRods(img, binarized_img, preliminary_connected_components, MAX_ROD_AREA)

    n_labels, label_ids, stats, centroids = \
        discardDistractorsAndArtifacts(cv.connectedComponentsWithStats(binarized_img, connectivity=4))

    cc_labeled_img = drawConnectedComponents(n_labels, label_ids, stats, centroids)

    rods_descriptions = []
    for label in range(1, n_labels):
        mask = getCCMask(label_ids, label)
        rods_descriptions.append(getRodTypeAndDescription(mask))

    mer_labeled_img = drawMERSWithAxes((n_labels, label_ids, stats, centroids), rods_descriptions)

    fig = plt.figure(figsize=(16, 16))

    fig.add_subplot(4, 4, 1)
    plt.imshow(img, cmap="gray")

    fig.add_subplot(4, 4, 2)
    plt.imshow(binarized_img, cmap="gray")

    fig.add_subplot(4, 4, 3)
    plt.imshow(cc_labeled_img)

    fig.add_subplot(4, 4, 4)
    plt.imshow(mer_labeled_img)

    plt.show()

    printRodDescriptions(rods_descriptions)
