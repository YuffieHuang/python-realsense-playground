import numpy as np
import cv2
import pyrealsense2 as rs
from pprint import pprint


class DepthRoiEvaluator(object):
    def __init__(self):
        pass

    @staticmethod
    def calc_center_of_mass(contour):
        epsilon = 0.1 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        c_x = np.array([c[0][0] for c in approx])
        c_y = np.array([c[0][1] for c in approx])
        return [np.mean(c_x), np.mean(c_y)]

    @staticmethod
    def calc_vertical_line(contour, x):
        _x, y, _w, h = cv2.boundingRect(contour)
        p1 = (int(x), y)
        p2 = (int(x), y+h)
        return p1, p2

    @staticmethod
    def calc_world_pos(x, y, depth_frame, tolerance_radius=5):
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        depth_candidates = []
        for x_i in range(-tolerance_radius, tolerance_radius):
            for y_i in range(-tolerance_radius, tolerance_radius):
                depth = depth_frame.get_distance(x+x_i, y+y_i)
                if depth < 10.0:
                    depth_candidates.append(depth)
        if len(depth_candidates) == 0:
            return None
        #print("candidates prior to filtering (len(", len(depth_candidates), ")")
        #pprint(depth_candidates)
        depth_candidates = np.array(depth_candidates)
        # reject outliers
        m = 2
        depth_candidates = depth_candidates[
            abs(depth_candidates - np.mean(depth_candidates)) < m * np.std(depth_candidates)]
        #print("candidates after filtering (len(", len(depth_candidates), ")")
        #pprint(depth_candidates)
        depth = np.mean(depth_candidates)
        #print("resulting depth", depth)
        return rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth), depth
