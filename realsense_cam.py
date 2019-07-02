import cv2
import pyrealsense2 as rs
import numpy as np


class RealsenseCam(object):
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

        self.colorizer = rs.colorizer(0)

    def __del__(self):
        self.pipeline.stop()

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()

        depth_frame = self.colorizer.colorize(frames.get_depth_frame())
        depth_image = np.asanyarray(depth_frame.get_data())

        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', depth_image)
        return jpeg.tobytes()