import pyrealsense2 as rs
import numpy as np
import cv2

if __name__ == "__main__":
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    alpha = 0.1
    while True:
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=alpha), cv2.COLORMAP_JET)

        cv2.imshow("ColorImage", color_image)
        cv2.imshow("DepthImage", depth_colormap)

        key = cv2.waitKey(1)

        if key == ord("q"):
            break
        elif key == ord("+"):
            alpha = min(alpha+0.01, 1.0)
        elif key == ord("-"):
            alpha = max(alpha-0.01, 0.01)

    pipeline.stop()
