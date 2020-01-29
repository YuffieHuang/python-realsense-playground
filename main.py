import pyrealsense2 as rs
import numpy as np
import cv2


def show_capture_feed():
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
            alpha = min(alpha + 0.01, 1.0)
        elif key == ord("-"):
            alpha = max(alpha - 0.01, 0.01)

    pipeline.stop()
    cv2.destroyAllWindows()


def selective_search(method="fast"):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # quality
    # speed-up using multithreads
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)

    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    # number of region proposals to show
    numShowRects = 100
    # increment to increase/decrease total number
    # of reason proposals to be shown
    method_set = False
    alpha = 0.1
    while True:
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        #color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        #color_image = np.asanyarray(color_frame.get_data())

        im = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=alpha), cv2.COLORMAP_JET)

        # resize image
        newHeight = 200
        newWidth = int(im.shape[1] * 200 / im.shape[0])
        im = cv2.resize(im, (newWidth, newHeight))
        # set input image on which we will run segmentation
        ss.setBaseImage(im)
        if method == "fast" and not method_set:
            ss.switchToSelectiveSearchFast()
            method_set = True
        elif method == "quality" and not method_set:
            ss.switchToSelectiveSearchQuality()
            method_set = True

        # run selective search segmentation on input image
        rects = ss.process()

        # iterate over all the region proposals
        for i, rect in enumerate(rects):
            # draw rectangle for region proposal till numShowRects
            if i < numShowRects:
                x, y, w, h = rect
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
            else:
                break

        # show output
        cv2.imshow("Output", im)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("+"):
            alpha = min(alpha + 0.01, 1.0)
        elif key == ord("-"):
            alpha = max(alpha - 0.01, 0.01)

    pipeline.stop()
    cv2.destroyAllWindows()


def get_labeled_img(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    return labeled_img


def connected_components():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    alpha = 0.1
    while True:
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        # color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        # color_image = np.asanyarray(color_frame.get_data())

        depth_thresh = cv2.threshold(cv2.convertScaleAbs(depth_image, alpha=alpha), 100, 255, cv2.THRESH_BINARY)[1]
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=alpha), cv2.COLORMAP_JET)

        ret, labels = cv2.connectedComponents(depth_thresh)

        img = get_labeled_img(labels)
        cv2.imshow("ConnectedComponents", img)
        cv2.imshow("DepthThresh", depth_thresh)
        cv2.imshow("DepthColormap", depth_colormap)

        key = cv2.waitKey(1)

        if key == ord("q"):
            break
        elif key == ord("+"):
            alpha = min(alpha + 0.01, 1.0)
        elif key == ord("-"):
            alpha = max(alpha - 0.01, 0.01)

    pipeline.stop()
    cv2.destroyAllWindows()


def blob_detector():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    params = cv2.SimpleBlobDetector_Params()
    #params.minThreshold = 100
    #params.maxThreshold = 700
    params.filterByArea = 1500

    detector = cv2.SimpleBlobDetector.create(params)

    alpha = 0.1
    while True:
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        # color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        # color_image = np.asanyarray(color_frame.get_data())

        depth_image = cv2.convertScaleAbs(depth_image, alpha=alpha)

        keypoints = detector.detect(depth_image)
        im = cv2.drawKeypoints(depth_image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow("Blobs", im)
        cv2.imshow("DepthImage", depth_image)

        key = cv2.waitKey(1)

        if key == ord("q"):
            break
        elif key == ord("+"):
            alpha = min(alpha + 0.01, 1.0)
        elif key == ord("-"):
            alpha = max(alpha - 0.01, 0.01)

    pipeline.stop()
    cv2.destroyAllWindows()


def canny_edge_detector():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    colorizer = rs.colorizer()

    alpha = 0.1
    run = True
    depth_frame = None
    draw_contour = -1
    while True:
        frames = pipeline.wait_for_frames()
        if run:
            depth_frame = colorizer.colorize(frames.get_depth_frame())
            #depth_frame = frames.get_depth_frame()

        #depth_image = cv2.convertScaleAbs(np.asanyarray(depth_frame.get_data()), alpha=0.1)
        depth_image = cv2.convertScaleAbs(np.asanyarray(depth_frame.get_data()), alpha=1.0)

        #depth_thresh = cv2.threshold(depth_image, np.average(depth_image), 255, 1)[1]
        depth_thresh = cv2.threshold(depth_image, 50, 255, 1)[1]
        imgray = cv2.cvtColor(depth_thresh, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)

        edges = cv2.Canny(depth_image, 100, 200, L2gradient=True)

        #ret, labels = cv2.connectedComponents(depth_image)
        #components = get_labeled_img(labels)

        #contours_img, contours, hierarchy = cv2.findContours(depth_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        """
        good_contours = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 50.0:
                good_contours.append(cnt)

        contours_img = np.zeros((contours_img.shape[0], contours_img.shape[1], 3), np.uint8)
        if draw_contour == -1:
            for i in range(0, len(good_contours)):
                M = cv2.moments(good_contours[i])
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                clr = (cX % 255, cY % 255, (cX+cY) % 255)
                blank_image = cv2.drawContours(contours_img, good_contours, i, clr, 3)
        else:
            blank_image = cv2.drawContours(contours_img, good_contours, draw_contour%len(good_contours), (0,125,0), 3)
        """

        #image = cv2.drawContours(blank_image, good_contours, -1, (0,255,0), 1) # -1 means filled, > 0 is line thickness



        cv2.imshow("DepthImage", depth_image)
        #cv2.imshow("Edges", edges)
        #cv2.imshow("connected comps", components)
        cv2.imshow("contours", contours_img)
        cv2.imshow("thresholded", depth_thresh)

        key = cv2.waitKey(1)

        if key == ord("q"):
            break
        elif key == ord("+"):
            alpha = min(alpha + 0.01, 1.0)
        elif key == ord("-"):
            alpha = max(alpha - 0.01, 0.01)
        elif key == ord("s"):
            run = not run
        elif key == ord("1"):
            draw_contour = -1
        elif key == ord("2"):
            draw_contour = 0
        elif key == ord("a") and draw_contour >= 0:
            draw_contour = max(draw_contour - 1, 0)
        elif key == ord("d") and draw_contour >= 0:
            draw_contour += 1

    pipeline.stop()
    cv2.destroyAllWindows()


def mser():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    colorizer = rs.colorizer(0)

    # Create MSER object
    mser = cv2.MSER_create(_min_area=200, _max_area=100000)

    alpha = 0.1
    while True:
        frames = pipeline.wait_for_frames()

        depth_frame = colorizer.colorize(frames.get_depth_frame())
        depth_image = np.asanyarray(depth_frame.get_data())

        depth_thresh = cv2.threshold(depth_image, 50, 255, 1)[1]

        depth_thresh = cv2.cvtColor(depth_thresh, cv2.COLOR_BGR2GRAY)
        # detect regions in gray scale image
        regions, _ = mser.detectRegions(depth_thresh)

        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

        mask = np.zeros((depth_image.shape[0], depth_image.shape[1], 1), dtype=np.uint8)
        cv2.polylines(mask, regions, 1, (255, 0, 0))

        cv2.imshow("DepthImage", depth_image)
        cv2.imshow("DepthThresh", depth_thresh)
        cv2.imshow("mask", mask)

        key = cv2.waitKey(1)

        if key == ord("q"):
            break
        elif key == ord("+"):
            alpha = min(alpha + 0.01, 1.0)
        elif key == ord("-"):
            alpha = max(alpha - 0.01, 0.01)

    pipeline.stop()
    cv2.destroyAllWindows()


depth_frame = None
color_frame = None


def frame_align():
    global depth_frame, color_frame

    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    config = rs.config()

    # This is the minimal recommended resolution for D435
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 90)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    colorizer = rs.colorizer(0)

    def hover_depth(event, x, y, flags, param):
        global depth_frame, color_frame
        if event == cv2.EVENT_MOUSEMOVE:
            print("distance at[", x, ",", y, "]", depth_frame.get_distance(x, y))

    depth_image_frame = "DepthImage"
    color_image_frame = "ColorImage"
    cv2.namedWindow(depth_image_frame)
    cv2.namedWindow(color_image_frame)
    cv2.setMouseCallback(depth_image_frame, hover_depth)
    cv2.setMouseCallback(color_image_frame, hover_depth)

    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        color_image = np.asanyarray(color_frame.get_data())

        cv2.imshow("DepthImage", depth_image)
        cv2.imshow("ColorImage", color_image)

        key = cv2.waitKey(1)

        if key == ord("q"):
            break
        elif key == ord("+"):
            alpha = min(alpha + 0.01, 1.0)
        elif key == ord("-"):
            alpha = max(alpha - 0.01, 0.01)

    pipeline.stop()
    cv2.destroyAllWindows()


def canny_2():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    colorizer = rs.colorizer(2)

    alpha = 0.1
    run = True
    depth_frame = None
    draw_contour = -1
    while True:
        frames = pipeline.wait_for_frames()
        if run:
            depth_frame = colorizer.colorize(frames.get_depth_frame())

        depth_image = cv2.convertScaleAbs(np.asanyarray(depth_frame.get_data()), alpha=1.0)
        depth_image = cv2.GaussianBlur(depth_image, (3, 3), cv2.BORDER_DEFAULT)

        edges = cv2.Canny(depth_image, 100, 200, L2gradient=True)

        cv2.imshow("DepthImage", depth_image)
        cv2.imshow("Edges", edges)

        key = cv2.waitKey(1)

        if key == ord("q"):
            break
        elif key == ord("+"):
            alpha = min(alpha + 0.01, 1.0)
        elif key == ord("-"):
            alpha = max(alpha - 0.01, 0.01)
        elif key == ord("s"):
            run = not run
        elif key == ord("1"):
            draw_contour = -1
        elif key == ord("2"):
            draw_contour = 0
        elif key == ord("a") and draw_contour >= 0:
            draw_contour = max(draw_contour - 1, 0)
        elif key == ord("d") and draw_contour >= 0:
            draw_contour += 1

    pipeline.stop()
    cv2.destroyAllWindows()


click_pos = [-1, -1]
crop_mask = None
depth_start = 10
depth_end = 10


def foreground_roi_depth_evaluation(measurement_height=0.125):
    global depth_frame, crop_mask, depth_start, depth_end
    from foreground_roi_detector import ForegroundRoiDetector
    from depth_roi_evaluator import DepthRoiEvaluator

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    def click_line(event, x, y, flags, param):
        global crop_mask, depth_frame
        if event == cv2.EVENT_LBUTTONDOWN:
            crop_mask = np.zeros_like(foreground_mask)
            cv2.rectangle(crop_mask, (depth_image.shape[1] - 1, y), (0, 0),
                          color=255, thickness=-1)
        if event == cv2.EVENT_MOUSEMOVE:
            print("distance at[", x, ",", y, "]", depth_frame.get_distance(x, y))

    color_image_frame = "ColorImage"
    cv2.namedWindow(color_image_frame)
    cv2.setMouseCallback(color_image_frame, click_line)

    colorizer = rs.colorizer(2)
    align_to = rs.stream.color
    align = rs.align(align_to)

    alpha = 0.1
    run = True
    depth_frame = None
    color_frame = None
    draw_contour = -1
    roi_detector = ForegroundRoiDetector()
    roi_evaluator = DepthRoiEvaluator()
    frames = None
    while True:
        if run:
            frames = pipeline.wait_for_frames()
            # Align the depth frame to color frame
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

        # extract depth and color images
        depth_image = cv2.convertScaleAbs(np.asanyarray(colorizer.colorize(depth_frame).get_data()), alpha=1.0)
        color_image = np.asanyarray(color_frame.get_data())

        # detect region of interest
        foreground_mask, foreground_contour = roi_detector.detect(depth_image=depth_image)
        if crop_mask is not None:
            cv2.imshow("CropMask", crop_mask)
            foreground_mask, foreground_contour = roi_detector.refine(foreground_mask, crop_mask, depth_image)

        # mask source images based on ROI
        original_color = color_image.copy()
        original_depth = depth_image.copy()
        #color_image = cv2.bitwise_and(color_image, color_image, mask=foreground_mask)
        depth_image = cv2.bitwise_and(depth_image, depth_image, mask=foreground_mask)

        # measure ROI diameter @ measurement height
        try:
            diameter = roi_evaluator.calc_diameter(foreground_contour, foreground_mask,
                                                   frames.get_depth_frame(),
                                                   measurement_height=measurement_height,
                                                   debug_img=color_image)
        except Exception as e:
            pass
        #print("diameter", diameter)

        # visualize result
        cv2.imshow("ForegroundMask", foreground_mask)
        cv2.imshow(color_image_frame, color_image)
        #cv2.imshow("Original Color", original_color)
        cv2.imshow("Original Depth", original_depth)

        key = cv2.waitKey(1)

        if key == ord("q"):
            break
        elif key == ord("+"):
            alpha = min(alpha + 0.01, 1.0)
        elif key == ord("-"):
            alpha = max(alpha - 0.01, 0.01)
        elif key == ord("s"):
            run = not run
        elif key == ord("1"):
            draw_contour = -1
        elif key == ord("2"):
            draw_contour = 0
        elif key == ord("a") and draw_contour >= 0:
            draw_contour = max(draw_contour - 1, 0)
        elif key == ord("d") and draw_contour >= 0:
            draw_contour += 1

    pipeline.stop()
    cv2.destroyAllWindows()


magnitude = None
t_line = None

def gradient_intensity_evaluation(measurement_height=0.125):
    global depth_frame, crop_mask, depth_start, depth_end, magnitude, t_line
    from depth_roi_evaluator import DepthRoiEvaluator
    import advanced_mode_example

    dev = advanced_mode_example.find_device_that_supports_advanced_mode()
    advnc_mode = advanced_mode_example.enter_advanced_mode(dev)
    depth_table = advnc_mode.get_depth_table()
    depth_table.depthClampMax = 3500
    depth_table.depthClampMin = 1000
    advnc_mode.set_depth_table(depth_table)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(dev.get_info(rs.camera_info.serial_number))
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    def click_line(event, x, y, flags, param):
        global crop_mask, depth_frame, magnitude, t_line
        if event == cv2.EVENT_LBUTTONDOWN:
            t_line = y
        if event == cv2.EVENT_MOUSEMOVE:
            pass
            #print("distance at[", x, ",", y, "]", depth_frame.get_distance(x, y))
            #print("magnitude at[", x, ",", y, "]", magnitude[y, x, 0])

    color_image_frame = "ColorImage"
    depth_image_frame = "DepthImage"
    depth_gradient_image_frame = "GradientImage"
    threshold_image_frame = "ThresholdImage"
    cv2.namedWindow(color_image_frame)
    cv2.namedWindow(color_image_frame)
    cv2.setMouseCallback(color_image_frame, click_line)

    colorizer = rs.colorizer(2)
    align_to = rs.stream.color
    align = rs.align(align_to)

    run = True
    depth_frame = None
    color_frame = None
    frames = None

    while True:
        if run:
            frames = pipeline.wait_for_frames()
            # Align the depth frame to color frame
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

        # extract depth and color images
        depth_image = cv2.convertScaleAbs(np.asanyarray(colorizer.colorize(depth_frame).get_data()), alpha=1.0)
        #depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        color_image = np.asanyarray(color_frame.get_data())

        """
        small_to_large_image_size_ratio = 0.5
        depth_image = cv2.resize(depth_image,  # original image
                               (0, 0),  # set fx and fy, not the final size
                               fx=small_to_large_image_size_ratio,
                               fy=small_to_large_image_size_ratio,
                               interpolation=cv2.INTER_NEAREST)
        depth_image = cv2.resize(depth_image,  # original image
                                 (0, 0),  # set fx and fy, not the final size
                                 fx=1/small_to_large_image_size_ratio,
                                 fy=1/small_to_large_image_size_ratio,
                                 interpolation=cv2.INTER_NEAREST)
        depth_image = cv2.GaussianBlur(depth_image, (11, 11), 0, cv2.BORDER_DEFAULT)
        """

        sobelx = cv2.Sobel(depth_image, cv2.CV_64F, 1, 0, ksize=11, scale=1, delta=0)  # Find x and y gradients
        sobely = cv2.Sobel(depth_image, cv2.CV_64F, 0, 1, ksize=11, scale=1, delta=0)

        # Find magnitude and angle
        magnitude = np.sqrt(sobelx ** 2.0 + sobely ** 2.0)
        angle = np.arctan2(sobely, sobelx) * (180 / np.pi)

        magnitude = cv2.normalize(magnitude, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        """
        height, width, channels = magnitude.shape
        for x in range(0, width):
            for y in range(0, height):
                print(magnitude[y, x])
        """

        # vertical line
        height, width, channels = color_image.shape
        p1 = (int(width / 2), height)
        if t_line is not None:
            p1 = (int(width / 2), t_line)
        p2 = (int(width/2), 0)
        line = (p1, p2)
        cv2.line(color_image, p1, p2, color=(0, 0, 255), thickness=2)

        # find measurement height line
        h = p1[1] - p2[1]
        sample_step = 1
        offset = 0
        pixel_end = None
        p_end = None
        p_end_fitness = -100000
        world_pos_res = DepthRoiEvaluator.calc_world_pos(p1[0], p1[1], depth_frame)
        p_start = None
        diff = 0
        depth_end = 0
        best_depth = 0
        if world_pos_res is not None:
            p_start, depth_start = world_pos_res
            while offset < h:
                offset += sample_step
                world_pos_res = DepthRoiEvaluator.calc_world_pos(
                    p1[0], p1[1] - offset, depth_frame, tolerance_radius=5)
                if world_pos_res is not None:
                    end_candidate, depth_end = world_pos_res
                    dir_vec = np.array(end_candidate) - np.array(p_start)
                    diff = np.linalg.norm(dir_vec)
                    fitness = 1 - abs(measurement_height - diff)
                    if fitness > p_end_fitness:
                        p_end = end_candidate
                        p_end_fitness = fitness
                        pixel_end = (p1[0], p1[1] - offset)
                        best_depth = depth_end
            #print(best_depth)
            #if pixel_end is not None:
            #    cv2.circle(color_image, pixel_end, radius=10, color=(255, 0, 0), thickness=-1)

        # find diameter line
        padding = 2
        pixel_left = None
        pixel_right = None
        if p_start is not None and p_end is not None:
            # find left diameter pixel
            offset = 0
            best_magnitude = 0
            while True:
                offset += sample_step
                curr_pixel = (pixel_end[0] - offset, pixel_end[1])
                out_of_bounds = curr_pixel[0] < 0 or curr_pixel[0] >= magnitude.shape[1]
                new_edge_found = False
                if not out_of_bounds:
                    new_edge_found = magnitude[curr_pixel[1], curr_pixel[0], 0] > 20
                stop = out_of_bounds or new_edge_found
                if stop:
                    if best_magnitude < magnitude[curr_pixel[1], curr_pixel[0], 0]:
                        best_magnitude = magnitude[curr_pixel[1], curr_pixel[0], 0]
                        pixel_left = (curr_pixel[0] + padding, curr_pixel[1])
                    else:
                        break
            # find right diameter pixel
            offset = 0
            best_magnitude = 0
            while True:
                offset += sample_step
                curr_pixel = (pixel_end[0] + offset, pixel_end[1])
                out_of_bounds = curr_pixel[0] < 0 or curr_pixel[0] >= magnitude.shape[1]
                new_edge_found = False
                if not out_of_bounds:
                    new_edge_found = magnitude[curr_pixel[1], curr_pixel[0], 0] > 20
                stop = out_of_bounds or new_edge_found
                if stop:
                    if (not out_of_bounds) and best_magnitude < magnitude[curr_pixel[1], curr_pixel[0], 0]:
                        best_magnitude = magnitude[curr_pixel[1], curr_pixel[0], 0]
                        pixel_right = (curr_pixel[0] - padding, curr_pixel[1])
                    else:
                        break

            if pixel_left is not None and pixel_right is not None:
                cv2.line(color_image, pixel_left, pixel_right, color=(0, 0, 255), thickness=2)
                cv2.line(depth_image, pixel_left, pixel_right, color=(0, 0, 255), thickness=2)
                cv2.line(magnitude, pixel_left, pixel_right, color=(0, 0, 255), thickness=2)
                wp_left, d_left = DepthRoiEvaluator.calc_world_pos(pixel_left[0], pixel_left[1],
                                                                   depth_frame, tolerance_radius=0)
                wp_right, d_right = DepthRoiEvaluator.calc_world_pos(pixel_right[0], pixel_right[1],
                                                                     depth_frame, tolerance_radius=0)
                dir_vec = np.array(wp_right) - np.array(wp_left)
                diff = np.linalg.norm(dir_vec)
                print("diameter is ", diff, "m")



        cv2.imshow(color_image_frame, color_image)
        cv2.imshow(depth_image_frame, depth_image)
        cv2.imshow(depth_gradient_image_frame, magnitude)

        key = cv2.waitKey(1)

        if key == ord("q"):
            break

    pipeline.stop()
    cv2.destroyAllWindows()


class RangeIndexMapping:
    def __init__(self, ranges=[]):
        self.ranges = ranges
        if len(self.ranges) == 0:
            self.ranges = [(7, 10), (10, 15), (15, 20)]

    def get_range_index(self, i):
        range_index = 0
        for r in self.ranges:
            if i in range(r[0], r[1]+1):
                return range_index
            range_index += 1
        return -1


class RangeIndexMatrix:
    def __init__(self, x_mapping=RangeIndexMapping(), y_mapping=RangeIndexMapping()):
        self.values = [[-1.0 for y_range in y_mapping.ranges] for x_range in x_mapping.ranges]
        self.x_mapping = x_mapping
        self.y_mapping = y_mapping

    def get_value(self, x_value, y_value):
        x_idx = self.x_mapping.get_range_index(x_value)
        y_idx = self.y_mapping.get_range_index(y_value)
        if x_idx != -1 and y_idx != -1:
            return self.values[x_idx][y_idx]


class Kiefer(RangeIndexMatrix):
    def __init__(self):
        super().__init__(
            x_mapping=RangeIndexMapping(ranges=[(7, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35)]), # heightclass
            y_mapping=RangeIndexMapping(ranges=[(0, 10), (10, 20), (20, 25), (30, 35), (40, 45), (50, 55), (60, 65), (65, 70)]) # diameter
        )
        self.values[0][0] = 0.34
        self.values[0][1] = 0.48
        self.values[1][1] = 0.45

    def get_formzahl(self, height_class, diameter):
        self.get_value(height_class, diameter)


def test_advanced():
    import advanced_mode_example
    advanced_mode_example.test_advanced_mode()


if __name__ == "__main__":
    #show_capture_feed()
    #selective_search("fast")
    #connected_components()
    #blob_detector()
    #canny_edge_detector()
    #mser()
    #frame_align()
    #canny_2()
    #foreground_roi_depth_evaluation(measurement_height=1.3)
    gradient_intensity_evaluation(measurement_height=0.5)
    #test_advanced()
