## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################
import sys
import pyrealsense2 as rs
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import cv2
import os
import time

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
# pipeline.start(config)
#
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)
#
times = 0
flag = 0
photo = 0
last_image=0
name = "Depth"
try:
    os.mkdir('../data_m2_to_m1/result/'+name)
except OSError:
    pass

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        # depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # now_image = np.sum(color_image[460:480][620:640])
        # print(color_image[460:480,620:640])
        # last_image = now_image

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.3), cv2.COLORMAP_JET)

        # Stack both images horizontally
        # images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # CHECK The LED position
        # images[450:480,610:640] = [255,255,255]

        cv2.imshow('RealSense', color_image)
        # if flag ==1:
            # cv2.imwrite('../data_m2_to_m1/bill23/f_%d.jpeg' % (times), color_image)
            # times+=1
        ############# Take picture ############

        for i in range(450,480):
            for j in range(610,640):
                if  color_image[i][j][2]>200 and color_image[i][j][0]<50 and color_image[i][j][1]<50:
                    flag = 1
                    if photo ==0:
                        print("Target Ready! Picture: ", times)
                        cv2.imwrite('../data_m2_to_m1/result/'+name+'/f_%d.jpeg' % (times), color_image)
                        # CROPf
                        # img = np.copy(images[:, 250:540])
                        # SAVE CROP IMAGES
                        # cv2.imwrite("../data_collection/rdm_data_0117/pictures_six_basic/out%d.jpeg" % times, images)
                        times += 1
                        photo=1
                        break
            if flag==1:
                break
        if flag == 0:
            photo= 0
        else:
            flag=0

        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()
