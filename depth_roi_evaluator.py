import numpy as np
import cv2
import pyrealsense2 as rs


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
    def is_in_bounds(x, y, frame):
        return not(x < 0 or x >= frame.width) and not(y < 0 or y >= frame.height)

    @staticmethod
    def calc_vertical_line(contour, x):
        _x, y, _w, h = cv2.boundingRect(contour)
        p1 = (int(x), y)
        p2 = (int(x), y+h)
        return p1, p2

    @staticmethod
    def calc_world_pos(x, y, depth_frame, tolerance_radius=5):
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        depth = 0
        if tolerance_radius > 0:
            depth_candidates = []
            for x_i in range(-tolerance_radius, tolerance_radius):
                for y_i in range(-tolerance_radius, tolerance_radius):
                    x_t = x+x_i
                    y_t = y+y_i
                    if not DepthRoiEvaluator.is_in_bounds(x_t, y_t, depth_frame):
                        continue
                    depth = depth_frame.get_distance(x_t, y_t)
                    if depth < 10.0:
                        depth_candidates.append(depth)
            if len(depth_candidates) == 0:
                return None
            depth_candidates = np.array(depth_candidates)
            # reject outliers
            m = 2
            depth_candidates = depth_candidates[
                abs(depth_candidates - np.mean(depth_candidates)) < m * np.std(depth_candidates)]
            depth = np.mean(depth_candidates)
        else:
            depth = depth_frame.get_distance(x, y)
        return rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth), depth

    @staticmethod
    def calc_diameter(foreground_contour, foreground_mask, depth_frame, measurement_height=0.125, debug_img=None):
        # calculate vertical line in center of mass
        center_of_mass = DepthRoiEvaluator.calc_center_of_mass(foreground_contour)
        line = DepthRoiEvaluator.calc_vertical_line(foreground_contour, center_of_mass[0])
        if debug_img is not None:
            cv2.line(debug_img, line[0], line[1], color=(0, 0, 255), thickness=2)
            cv2.drawContours(debug_img, [foreground_contour], 0, color=(0, 255, 0), thickness=2)
        p_start, depth_start = DepthRoiEvaluator.calc_world_pos(line[1][0], line[1][1], depth_frame)
        h = line[1][1] - line[0][1]

        # find measurement height line
        sample_step = 1
        offset = 0
        pixel_end = None
        p_end = None
        p_end_fitness = -100000
        while offset < h:
            offset += sample_step
            end_candidate, depth_end = DepthRoiEvaluator.calc_world_pos(
                line[1][0], line[1][1] - offset, depth_frame)
            dir_vec = np.array(end_candidate) - np.array(p_start)
            diff = np.linalg.norm(dir_vec)
            fitness = 1 - abs(measurement_height - diff)
            if fitness > p_end_fitness:
                p_end = end_candidate
                p_end_fitness = fitness
                pixel_end = (line[1][0], line[1][1] - offset)

        # find diameter line
        padding = 2
        pixel_left = None
        pixel_right = None
        if p_start is not None and p_end is not None:
            if debug_img is not None:
                cv2.circle(debug_img, pixel_end, radius=10, color=(255, 0, 0), thickness=-1)
            # find left diameter pixel
            offset = 0
            while True:
                offset += sample_step
                curr_pixel = (pixel_end[0] - offset, pixel_end[1])
                stop = curr_pixel[0] < 0 or curr_pixel[0] >= foreground_mask.shape[1] or \
                       foreground_mask[curr_pixel[1], curr_pixel[0]] == 0
                if stop:
                    if pixel_left is not None:
                        pixel_left = (pixel_left[0] + padding, pixel_left[1])
                    break
                pixel_left = curr_pixel
            # find right diameter pixel
            offset = 0
            while True:
                offset += sample_step
                curr_pixel = (pixel_end[0] + offset, pixel_end[1])
                stop = curr_pixel[0] < 0 or curr_pixel[0] >= foreground_mask.shape[1] or \
                       foreground_mask[curr_pixel[1], curr_pixel[0]] == 0
                if stop:
                    if pixel_right is not None:
                        pixel_right = (pixel_right[0] - padding, pixel_right[1])
                    break
                pixel_right = curr_pixel

        # calculate diameter
        diameter = -1
        if pixel_left is not None and pixel_right is not None:
            p_left, depth_left = DepthRoiEvaluator.calc_world_pos(
                pixel_left[0], pixel_left[1],
                depth_frame,
                tolerance_radius=0
            )
            p_right, depth_right = DepthRoiEvaluator.calc_world_pos(
                pixel_right[0], pixel_right[1],
                depth_frame,
                tolerance_radius=0
            )

            dir_vec = np.array(p_right) - np.array(p_left)
            diameter = np.linalg.norm(dir_vec)

            if debug_img is not None:
                cv2.line(debug_img, pixel_left, pixel_right, color=(0, 0, 255), thickness=2)

        return diameter
