#!/usr/bin/env python

import roslib
import numpy as np
import matplotlib.pyplot as plt
import os.path
import scipy
import argparse
import math
import cv2
import sys
import time
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
# Instantiate CvBridge
bridge = CvBridge()

sys.path.append('/usr/local/lib/python2.7/site-packages')
# Make sure that caffe is on the python path:
caffe_root = '/home/phenom/phenom/caffe-segnet/'
sys.path.insert(0, caffe_root + 'python')
import caffe
pub = rospy.Publisher('segnet', Image, queue_size=10)


#cv2.namedWindow("Input")
#cv2.namedWindow("Output")
#cap = cv2.VideoCapture("/home/phenom/phenom/caffe-segnet/SegNet-Tutorial/Scripts/1.mp4") # Change this to your webcam ID, or file name for your video file
fourcc = cv2.cv.CV_FOURCC(*'DIVX')
out_video = cv2.VideoWriter('output22.avi',fourcc, 20.0, (960, 360))
#rval = True

def segmentation(frame, net, input_shape):

	frame = cv2.resize(frame, (input_shape[3],input_shape[2]))
	input_image = frame.transpose((2,0,1))
	# input_image = input_image[(2,1,0),:,:] # May be required, if you do not open your data with opencv
	input_image = np.asarray([input_image])

	out = net.forward_all(data=input_image)

	#start = time.time()
	segmentation_ind = np.squeeze(net.blobs['argmax'].data)
	segmentation_ind_3ch = np.resize(segmentation_ind,(3,input_shape[2],input_shape[3]))
	segmentation_ind_3ch = segmentation_ind_3ch.transpose(1,2,0).astype(np.uint8)
	segmentation_rgb = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)

	cv2.LUT(segmentation_ind_3ch,label_colours,segmentation_rgb)
	#segmentation_rgb = segmentation_rgb.astype(float)/255

	#end = time.time()
	#print '%30s' % 'Processed results in ', str((end - start)*1000), 'ms\n'

	#cv2.imshow("Input", frame)
	#cv2.imshow("Output", segmentation_rgb)
	
	
	#frame_float = segmentation_rgb.astype(np.uint8)
	#cv2.imshow("Input_float", frame_float)
	#print(frame.dtype, segmentation_rgb.dtype, frame_float.dtype)
	vis = np.concatenate((frame, segmentation_rgb), axis=1)
	out_video.write(vis)
	#cv2.imshow("Output", vis)
	#print(str(frame.shape) + " " + str(segmentation_rgb.shape))	
	#print(vis.shape)
	# key = cv2.waitKey(1)
	# if key == 27: # exit on ESC
	#     break  
	return segmentation_rgb 

def image_callback(msg):
	global pub
	print("Received an image!")
	try:
        # Convert your ROS Image message to OpenCV2
	    cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
	except CvBridgeError, e:
	    print(e)
	else:
        # Save your OpenCV2 image as a jpeg 
	    segmentation_rgb = segmentation(cv2_img, net, input_shape)
	    image_message = bridge.cv2_to_imgmsg(segmentation_rgb, encoding="passthrough")
	    pub.publish(image_message)

if __name__ == '__main__':
	# model definitions
	net = caffe.Net('/home/phenom/phenom/catkin_ws/src/perception/segmentation/SegNet-Tutorial/Example_Models/segnet_model_driving_webdemo.prototxt',
	                '/home/phenom/phenom/catkin_ws/src/perception/segmentation/SegNet-Tutorial/Example_Models/segnet_weights_driving_webdemo.caffemodel',
	                caffe.TEST)

	caffe.set_mode_gpu()
	input_shape = net.blobs['data'].data.shape
	output_shape = net.blobs['argmax'].data.shape
	label_colours = cv2.imread('/home/phenom/phenom/caffe-segnet/SegNet-Tutorial/Scripts/camvid12.png').astype(np.uint8)	
	rospy.init_node('segmentation')
	rate = rospy.Rate(10) # 10hz

	while not rospy.is_shutdown():
	    # Define your image topic
	    image_topic = "/usb_cam/image_raw"
	    # Set up your subscriber and define its callback
	    rospy.Subscriber(image_topic, Image, image_callback)
	    # Spin until ctrl + c
	    rate.sleep()
	    rospy.spin()

	# cap.release()
	# out_video.release()
	# cv2.destroyAllWindows()

