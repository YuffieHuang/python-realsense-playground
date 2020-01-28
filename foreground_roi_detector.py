import cv2
import numpy as np
from enum import Enum


class Method(Enum):
    MIN_MEAN_DISTANCE = 0
    MIN_DISTANCE_TO_CENTER = 1


class ForegroundRoiDetector(object):
    def __init__(self, method=Method.MIN_MEAN_DISTANCE, min_area=5000, blur=True):
        self.method = method
        self.min_area = min_area
        self.blur = blur

    def detect(self, depth_image):
        contours = ForegroundRoiDetector.detect_contours(
            depth_image=depth_image,
            min_area=self.min_area,
            blur=self.blur
        )
        c_idx = self.calc_dominant_contour_index(contours, depth_image)
        mask = ForegroundRoiDetector.calc_contour_mask(depth_image.shape, contours, c_idx)
        contour = None
        if c_idx >= 0:
            contour = contours[c_idx]
        return mask, contour

    def refine(self, foreground_mask, crop_mask, depth_image):
        foreground_mask = cv2.bitwise_and(foreground_mask, foreground_mask, mask=crop_mask)
        _, contours, __ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c_idx = self.calc_dominant_contour_index(contours, depth_image=depth_image)
        contour = None
        if c_idx >= 0:
            contour = contours[c_idx]
        foreground_mask = self.calc_contour_mask(depth_image.shape, contours, c_idx)
        return foreground_mask, contour

    def calc_dominant_contour_index(self, contours, depth_image):
        dominant_contour = -1
        if self.method == Method.MIN_MEAN_DISTANCE:
            return self.calc_dominant_contour_index_min_mean_distance_based(contours, depth_image)
        elif self.method == Method.MIN_DISTANCE_TO_CENTER:
            return self.calc_dominant_contour_index_min_distance_to_center_based(contours, depth_image)
        return dominant_contour

    @staticmethod
    def calc_dominant_contour_index_min_mean_distance_based(contours, depth_image):
        dominant_contour = -1
        max_mean = -1
        for i in range(len(contours)):
            img = np.zeros_like(depth_image)
            cv2.drawContours(img, contours, i, color=255, thickness=-1)
            pts = np.where(img == 255)
            intensity_mean = np.mean(depth_image[pts[0], pts[1]])
            if max_mean < intensity_mean:
                dominant_contour = i
                max_mean = intensity_mean
        return dominant_contour

    @staticmethod
    def calc_dominant_contour_index_min_distance_to_center_based(contours, depth_image):
        dominant_contour = -1
        img_cx = depth_image.shape[1] / 2.0
        img_cy = depth_image.shape[0] / 2.0
        min_dist = 100000000000000000
        for i in range(len(contours)):
            M = cv2.moments(contours[i])
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            dist = np.linalg.norm([img_cx - cx, img_cy - cy])
            if dist < min_dist:
                dominant_contour = i
                min_dist = dist
        return dominant_contour

    @staticmethod
    def calc_contour_mask(shape, contours, c_idx):
        mask = np.zeros((shape[0], shape[1], 1), dtype=np.uint8)
        cv2.drawContours(mask, contours, c_idx, color=255, thickness=-1)
        return mask

    @staticmethod
    def detect_contours(depth_image, min_area=5000, blur=True):
        # gaussian blur to smooth edges and avoid tiny region due to noisy image
        if blur:
            depth_image = cv2.GaussianBlur(depth_image, (3, 3), cv2.BORDER_DEFAULT)
        # calc binarized image
        ret, thresh = cv2.threshold(cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY), 127, 255, 0)
        cv2.imshow("Initial Thresh", thresh)
        # compute contours (include only root regions)
        # RETR_TREE to retrieve all contours
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # return results based on min area filter
        return [c for c in contours if cv2.contourArea(c) > min_area]
