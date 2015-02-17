import cv2
import random
import numpy as np
import copy
import sys
import os
import caffe
import skimage

def draw_arrow(image, start, length, angle):
	print angle
	x_ = int(- np.sin(angle) * (-length) + start[0])
	y_ = int(np.cos(angle) * (-length) - start[1])

	x1_ = int((np.cos(angle) * (-length*0.4) - np.sin(angle) * (-length*0.6)) + start[0])
	y1_ = int((np.sin(angle) * (-length*0.4) + np.cos(angle) * (-length*0.6)) - start[1])
	x2_ = int((np.cos(angle) * (length*0.4) - np.sin(angle) * (-length*0.6)) + start[0])
	y2_ = int((np.sin(angle) * (length*0.4) + np.cos(angle) * (-length*0.6)) - start[1])

	# print start, (x_, -y_)

	cv2.line(image, start, (x_, -y_), (0, 0, 255), 8)
	cv2.line(image, (x_, -y_), (x1_, -y1_), (0,0,255), 5)
	cv2.line(image, (x_, -y_), (x2_, -y2_), (0,0,255), 5)

	# cv2.circle(image, (x_, y_), int(length/8), (0,0,255), 3)



def draw_ellipse(image, start, length, angle):
	cv2.ellipse(image, start, (80,80), 0, angle-112, angle-68, (0,0,200), -5)
	print angle-112, angle-68

def get_image(c):
	ret, frame = c.read()
	return frame 

def head_pose_estimation(headimage_, image, location_):
	# Set the right path to your model definition file, pretrained model weights,
	# and the image you would like to classify.
	MODEL_FILE = 'deploy_final'
	PRETRAINED = 'model_final2'
	# IMAGE_FILE = caffe_root + 'examples/images/cat.jpg'

	head_image = []
	for head in headimage_:
		head_image.append(skimage.img_as_float(head).astype(np.float32))
	# cv2.imshow("head",head_image)
	# cv2.imwrite("head/head"+str(frame).zfill(4)+".jpg", headimage_)
	net = caffe.Classifier(MODEL_FILE, PRETRAINED,
	                       mean=np.load('mean.npy'),
	                       image_dims=(32,32),
	                       raw_scale=255,
	                       gpu=True)

	net.set_phase_test()

	prediction = net.predict(head_image, oversample=False)
	# prediction[:,[0,1]] = prediction[:,[1,0]]
	# # print prediction

	for i, location_i in enumerate(location_):

		predict_class = np.argmax(prediction[i])
		# print prediction[predict_max][1:]
		# predict_class = prediction[predict_max][1:].argmin()
		# print predict_class

		# if predict_class == 0:
		# 	angle = 180
		# elif predict_class == 1:
		# 	angle = 0
		# elif predict_class == 2:
		# 	angle = 270
		# elif predict_class == 3:
		# 	angle = 90
		# else:
		# 	angle = 45
		angle = predict_class * 45
		print angle
		location = (int(location_i[0]), int(location_i[1])) 
		draw_arrow(image, location, 80, angle * 2.0 * np.pi / 360)



	# fontface = cv2.FONT_HERSHEY_DUPLEX 
	# fontscale = 2.0
	# color = (255, 255, 255)
	# location = (int(location_[0]-10), int(location_[1]-10)) 

	# msg = str(predict_angle)

	# if predict_class == 0:
	# 	msg = "BACK"
	# elif predict_class == 1:
	# 	msg = "FRONT"
	# elif predict_class == 2:
	# 	msg = "LEFT"
	# elif predict_class == 3:
	# 	msg = "RIGHT"
	# else:
	# 	msg = "???"

	# cv2.putText(image, msg, location, fontface, fontscale, color)

"""
for module
"""
def classify_byimage(detect_mfile, image_file, work_dir):

	# output.txt
	detections = open(detect_mfile, "r")

	detections_array = []
	for line in detections.readlines():
		detections_array.append(line.split())

	image = cv2.imread(image_name)
	height = image.shape[0]
	width = image.shape[1]

	head_image = []
	location = []
	candidate_num = len(detections_array)
	candidate_array = [[0 for i in xrange(8)] for i in xrange(candidate_num)]

	for c, candidate in enumerate(detections_arrays):
		for i in xrange(8):
			candidate_array[c][i] = int(candidate[i])

		cv2.rectangle(image, (candidate_array[c][4], candidate_array[c][5]), (candidate_array[c][4] + candidate_array[c][6], candidate_array[c][5] + candidate_array[c][7]), (255, 0, 0), 3)
		
		if candidate_array[c][4] > 0 and candidate_array[c][5] > 0 and candidate_array[c][4] + candidate_array[c][6] < width and candidate_array[c][5] + candidate_array[c][7] < height:
			head_image.append(image[candidate_array[c][5]:candidate_array[c][5]+candidate_array[c][7], candidate_array[c][4]:candidate_array[c][4]+candidate_array[c][6]])
			location.append([candidate_array[c][4], candidate_array[c][5]])
	
	head_pose_estimation(head_image, image, location)
	cv2.imwrite(work_dir + "/" + os.path.basename(imagename), image)

def main():
	filenames = []

	# get images
	for root, dirs, files in os.walk("/home/hokutokagaya/exp/detect_heads/dpm_test"):
		for file_ in files:
			filenames.append(os.path.join(root, file_))

	filenames.sort()
	print filenames

	# output.txt
	detections = open("output.txt", "r")

	detections_arrays = [[] for i in xrange(1500)]
	for line in detections.readlines():
		if not line.split()[1:] in detections_arrays[int(line.split()[0])]: 
			detections_arrays[int(line.split()[0])].append(line.split()[1:])

	image = cv2.imread(filenames[0])
	height = image.shape[0]
	width = image.shape[1]

	frame_num = 1

	for imagename in filenames:

		# print frame_num
		# cv2.waitKey(0)

		image = cv2.imread(imagename)
		
		head_image = []
		location = []
		candidate_num = len(detections_arrays[frame_num])
		candidate_array = [[0 for i in xrange(8)] for i in xrange(candidate_num)]
		for c, candidate in enumerate(detections_arrays[frame_num]):
			for i in xrange(8):
				candidate_array[c][i] = int(candidate[i])

			cv2.rectangle(image, (candidate_array[c][4], candidate_array[c][5]), (candidate_array[c][4] + candidate_array[c][6], candidate_array[c][5] + candidate_array[c][7]), (255, 0, 0), 3)
			
			if candidate_array[c][4] > 0 and candidate_array[c][5] > 0 and candidate_array[c][4] + candidate_array[c][6] < width and candidate_array[c][5] + candidate_array[c][7] < height:
				head_image.append(image[candidate_array[c][5]:candidate_array[c][5]+candidate_array[c][7], candidate_array[c][4]:candidate_array[c][4]+candidate_array[c][6]])
				location.append([candidate_array[c][4]+candidate_array[c][6]/2, candidate_array[c][5]+candidate_array[c][7]/2])
		
		head_pose_estimation(head_image, image, location)
		

	

		# cv2.imshow(window_name, image) 

		# image = cv2.resize(image, (int(width/2), int(height/2)))
		# cv2.imshow(window_name, image)
		cv2.imwrite("image_demo2/"+os.path.basename(imagename), image)
		frame_num += 1





if __name__ == '__main__':
	main()