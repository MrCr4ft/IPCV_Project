import cv2 as cv
import numpy as np


def binarizeImage(image: np.ndarray) -> np.ndarray:
    blurred_img = cv.boxFilter(src=image, ddepth=-1, ksize=(3,3))
    threshold, binarized_image = cv.threshold(blurred_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return threshold, 255 - binarized_image
