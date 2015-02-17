import sys
import cv2
import caffe, caffe.io
import skimage
import numpy as np
import os
import time
import csv


def main():
	if len(sys.argv) < 2:
		"Usage: detect_head_from_image.py image_dir [output_dir]"
		sys.exit()
	else:
		image_dir = sys.argv[1]

	# Define and initialize models
	MODEL_FILE = 'qmul_head_detect_deploy'
	PRETRAINED = 'qmul_head_detect_2'

	BMODEL_FILE = 'upperbody_detect_deploy'
	BPRETRAINED = 'upperbody_detect'

	net = caffe.Classifier(MODEL_FILE, PRETRAINED,
	                       mean=np.load('mean.npy'),
	                       image_dims=(32,32),
	                       raw_scale=255,
	                       gpu=True)

	bnet = caffe.Classifier(BMODEL_FILE, BPRETRAINED,
							mean=np.load('umean.npy'),
							image_dims=(32,32),
							raw_scale=255,
							gpu=True)

	net.set_phase_test()
	net.set_mode_gpu()
	bnet.set_phase_test()
	bnet.set_mode_gpu()

	# Read ground truth file
	top_file = sys.argv[2]
	top = open(top_file, 'rb')
	reader = csv.reader(top)

	top_csv = []
	images_info = []

	for row in reader:
		top_csv.append(row)

	for data in top_csv:
		if data[0] == image_dir:
			images_info.append(data)
	
	# Read images
	images = []

	for root, dirs, files in os.walk(image_dir):
		for file_ in files:
			filename = os.path.join(root, file_)
			images.append(filename)

	images.sort()

	# semi-global variables and constants
	results = []
	output_index = 1
	DETECT_WINDOW_SIZE = [40, 35, 30]


	# detect by image
	for filename in images:
		time1 = time.clock()
		print 'Now, {}'.format(filename)
		
		image = cv2.imread(filename)
		image2 = image + 0

		if image == None:
			print 'invalid file'
			continue

		height, width = image.shape[:2]
		
		"""
		Upper body detection!
		"""

		# target_queue = []
		# target_info_queue = []

		# target_result_history = np.array([[0,0]])
		# target_info_history = []
		

		# # Region for sliding windows
		# height_search_start = 0
		# width_search_start = 0

		# max_conf = -10.0
		# detected_face = None

		# # for debug variable
		# count = 0

		# DETECT_WINDOW_SIZE = [int(height/3)]
		# print DETECT_WINDOW_SIZE

		# for size in DETECT_WINDOW_SIZE:
			
		# 	height_search_end = height / 2 - size
		# 	width_search_end = width - size
		# 	for j in xrange(height_search_start, height_search_end, 2):
		# 		for i in xrange(width_search_start, width_search_end, 2):
		# 			count += 1
					
		# 			target_ = image[j:j+size, i:i+size]

		# 			target = skimage.img_as_float(target_).astype(np.float32)
		# 			target_queue.append(target)
		# 			target_info_queue.append([i, j, size])
		# 			target_info_history.append([i, j, size])

		# 			# prediction = net.predict([target], oversample=False)

		# 			# if prediction[0][1] > max_conf:
		# 			# 	detected_face = {"x":i, "y":j, "size":size}
		# 			# 	max_conf = prediction[0][1]

		# 			if len(target_queue) == 400:
		# 				prediction = net.predict(target_queue, oversample=False)
		# 				# print type(prediction), prediction[:,1].max()
		# 				local_max = prediction[:,1].max()
		# 				local_max_idx = prediction[:,1].argmax()

		# 				if local_max > max_conf:
		# 					mi, mj, msize = target_info_queue[local_max_idx]
		# 					detected_face = {"x":mi, "y":mj, "size":msize, "conf":local_max}
		# 					max_conf = local_max


		# 	# print detected_face


		# if len(target_queue) != 0:
		# 	prediction = net.predict(target_queue, oversample=False)
		# 	# print type(prediction), prediction[:,1].max()
		# 	local_max = prediction[:len(target_queue),1].max()
		# 	local_max_idx = prediction[:len(target_queue),1].argmax()

		# 	if local_max > max_conf:
		# 		mi, mj, msize = target_info_queue[local_max_idx]
		# 		detected_face = {"x":mi, "y":mj, "size":msize, "conf":local_max}
		# 		max_conf = local_max

		# 	target_result_history = np.vstack((target_result_history, prediction))

		# 	target_queue = []
		# 	target_info_queue = []


		# image2 = image + 0

		# target_result_history = target_result_history[1:, 1]
		# rank = np.argsort(target_result_history)
		# target_info_history = np.array(target_info_history)

		# # Green indicates MAX i, j, size
		# min_i = 10000
		# min_j = 10000
		# max_size = 0
		# max_i = 0
		# max_j = 0
		
		# if len(rank) > 5:
		# 	for i in xrange(1, 6):
		# 		info = target_info_history[rank[-i]]
		# 		if info[0] < min_i:
		# 			min_i = info[0]
		# 		if info[1] < min_j:
		# 			min_j = info[1]
		# 		if info[2] > max_size:
		# 			max_size = info[2]
			
		# 	# if info[0] + info[2] > max_i:
		# 	# 	max_i = info[0] + info[2]
		# 	# if info[1] + info[2] > max_j:
		# 	# 	max_j = info[1] + info[2]

		# 	# if max_j - min_j > max_i - min_i:
		# 	# 	max_size = max_j - min_j
		# 	# else:
		# 	# 	max_size = max_i - min_i


		# 	cv2.rectangle(image2, (min_i, min_j), (min_i+max_size, min_j+max_size), (0, 255, 0))

		# 	# # Red indicate AVG
		# 	# mean = np.array([0.0, 0.0, 0.0])
		# 	# for i in xrange(1, 6):
		# 	# 	mean = mean + np.array(target_info_history[rank[-i]]) / 5.0
				
		# 	# mean = mean.astype(np.int64)

		# 	# # print (mean[0], mean[1]), (mean[0]+mean[2], mean[1]+mean[2])
		# 	# cv2.rectangle(image2, (mean[0], mean[1]), (mean[0]+mean[2], mean[1]+mean[2]), (0, 0, 255))			
			
		# 	# Blue indicate MAX
		# 	# print (detected_face['x'], detected_face['y']), (detected_face['x']+detected_face['size'], detected_face['y']+detected_face['size'])
		# 	# cv2.rectangle(image2, (detected_face['x'], detected_face['y']), (detected_face['x']+detected_face['size'], detected_face['y']+detected_face['size']), (255, 0, 0))

		# else:
		# 	print 'no face', len(rank)



		"""
		Head detection!
		"""
		target_queue = []
		target_info_queue = []

		target_result_history = np.array([[0,0]])
		target_info_history = []

		# Region for sliding windows
		height_search_start = 0
		width_search_start = int(width / 6)

		max_conf = -10.0
		detected_face = None

		# for debug variable
		count = 0

		DETECT_WINDOW_SIZE = [int(width*1.3/3), int(width*1.1/3), int(width*0.9/3)]
		print 'WINDOW SIZE = ', DETECT_WINDOW_SIZE


		for size in DETECT_WINDOW_SIZE:
			
			height_search_end = height / 3 - size
			width_search_end = width * 5 / 6 - size
			for j in xrange(height_search_start, height_search_end, 3):
				for i in xrange(width_search_start, width_search_end, 3):
					count += 1
					
					target_ = image[j:j+size, i:i+size]

					target = skimage.img_as_float(target_).astype(np.float32)
					target_queue.append(target)
					target_info_queue.append([i, j, size])
					target_info_history.append([i, j, size])

					# prediction = net.predict([target], oversample=False)

					# if prediction[0][1] > max_conf:
					# 	detected_face = {"x":i, "y":j, "size":size}
					# 	max_conf = prediction[0][1]

					if len(target_queue) == 400:
						prediction = net.predict(target_queue, oversample=False)
						# print type(prediction), prediction[:,1].max()
						local_max = prediction[:,1].max()
						local_max_idx = prediction[:,1].argmax()
						target_result_history = np.vstack((target_result_history, prediction))

						# if local_max > max_conf:
						# 	mi, mj, msize = target_info_queue[local_max_idx]
						# 	detected_face = {"x":mi, "y":mj, "size":msize, "conf":local_max}
						# 	max_conf = local_max

						target_queue = []
						target_info_queue = []



			# print detected_face


		if len(target_queue) != 0:
			prediction = net.predict(target_queue, oversample=False)
			# print type(prediction), prediction[:,1].max()
			local_max = prediction[:len(target_queue),1].max()
			local_max_idx = prediction[:len(target_queue),1].argmax()

			if local_max > max_conf:
				mi, mj, msize = target_info_queue[local_max_idx]
				detected_face = {"x":mi, "y":mj, "size":msize, "conf":local_max}
				max_conf = local_max

			target_result_history = np.vstack((target_result_history, prediction))

			target_queue = []
			target_info_queue = []


		target_result_history = target_result_history[1:, 1]
		rank = np.argsort(target_result_history)

		target_info_history = np.array(target_info_history)


		# Black indicate ground-truth
		xs = int(float(images_info[output_index - 1][4]) - float(images_info[output_index - 1][8]))
		ys = int(float(images_info[output_index - 1][5]) - float(images_info[output_index - 1][9]))
		xe = int(float(images_info[output_index - 1][6]) - float(images_info[output_index - 1][8]))
		ye = int(float(images_info[output_index - 1][7]) - float(images_info[output_index - 1][9]))
		cv2.rectangle(image2, (xs, ys), (xe, ye), (0, 0, 0))

		for it, target_ in enumerate(target_result_history):
			if target_ > 0.8:
				x = target_info_history[it][0]
				y = target_info_history[it][1]
				size = target_info_history[it][2]
				cv2.rectangle(image2, (x, y), (x+size, y+size), (255, 0, 0))




		


		# Green indicates MAX i, j, size
		min_i = 10000
		min_j = 10000
		max_size = 0
		max_i = 0
		max_j = 0
		
		if len(rank) > 5:
			for i in xrange(1, 6):
				info = target_info_history[rank[-i]]
				# if info[0] < min_i:
				# 	min_i = info[0]
				if info[1] < min_j:
					min_i = info[0]
					min_j = info[1]
				if info[2] > max_size:
					max_size = info[2]

			# if info[0] + info[2] > max_i:
			# 	max_i = info[0] + info[2]
			# if info[1] + info[2] > max_j:
			# 	max_j = info[1] + info[2]

			# if max_j - min_j > max_i - min_i:
			# 	max_size = max_j - min_j
			# else:
			# 	max_size = max_i - min_i


			cv2.rectangle(image2, (min_i, min_j), (min_i+max_size, min_j+max_size), (0, 255, 0))

			# Red indicate AVG
			# mean = np.array([0.0, 0.0, 0.0])
			# for i in xrange(1, 6):
			# 	mean = mean + np.array(target_info_history[rank[-i]]) / 5.0
				
			# mean = mean.astype(np.int64)

			# # print (mean[0], mean[1]), (mean[0]+mean[2], mean[1]+mean[2])
			# cv2.rectangle(image2, (mean[0], mean[1]), (mean[0]+mean[2], mean[1]+mean[2]), (0, 0, 255))			
			
			# Blue indicate MAX
			# print (detected_face['x'], detected_face['y']), (detected_face['x']+detected_face['size'], detected_face['y']+detected_face['size'])
			# cv2.rectangle(image2, (detected_face['x'], detected_face['y']), (detected_face['x']+detected_face['size'], detected_face['y']+detected_face['size']), (255, 0, 0))

		else:
			print 'no face', len(rank)

		# Output 
		image2 = cv2.resize(image2, (100, 200))
		cv2.imwrite('result_image/{}r/test4_{}_{}.jpg'.format(image_dir, image_dir, str(output_index).zfill(4)), image2)
		results.append(cv2.resize(image2, (50, 120)))

		time2 = time.clock()

		print "Now computing time : {} sec".format(str(int(time2-time1))), height, width, max_size
		output_index += 1
		

	# grobal output
	rs = cv2.hconcat(results)

	cv2.imwrite("result_image/result4_{}.jpg".format(image_dir), rs)







if __name__ == '__main__':
	main()