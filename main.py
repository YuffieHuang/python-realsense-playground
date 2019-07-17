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
        color_image = cv2.bitwise_and(color_image, color_image, mask=foreground_mask)
        depth_image = cv2.bitwise_and(depth_image, depth_image, mask=foreground_mask)

        # measure ROI diameter @ measurement height
        diameter = roi_evaluator.calc_diameter(foreground_contour, foreground_mask,
                                               frames.get_depth_frame(),
                                               measurement_height=measurement_height,
                                               debug_img=color_image)
        print("diameter", diameter)

        # visualize result
        cv2.imshow("ForegroundMask", foreground_mask)
        cv2.imshow("DepthImage", depth_image)
        cv2.imshow(color_image_frame, color_image)

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


if __name__ == "__main__":
    #show_capture_feed()
    #selective_search("fast")
    #connected_components()
    #blob_detector()
    #canny_edge_detector()
    #mser()
    #frame_align()
    #canny_2()
    foreground_roi_depth_evaluation(measurement_height=0.125)
