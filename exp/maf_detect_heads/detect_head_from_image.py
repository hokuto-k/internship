import sys
import cv2
import caffe, caffe.io
import skimage
import numpy as np
import os
import time


def main():
	if len(sys.argv) < 2:
		"Usage: detect_head_from_image.py image_dir [output_dir]"
		sys.exit()
	else:
		image_dir = sys.argv[1]

	MODEL_FILE = 'qmul_head_detect_deploy'
	PRETRAINED = 'qmul_head_detect'

	net = caffe.Classifier(MODEL_FILE, PRETRAINED,
	                       mean=np.load('mean.npy'),
	                       image_dims=(32,32),
	                       raw_scale=255,
	                       gpu=True)

	net.set_phase_test()
	net.set_mode_gpu()

	DETECT_WINDOW_SIZE = [40, 35, 30, 25, 20]
	
	results = []
	images = []

	for root, dirs, files in os.walk(image_dir):
		for file_ in files:
			filename = os.path.join(root, file_)
			images.append(filename)

	images.sort()

	for filename in images:
		time1 = time.clock()
		print 'Now, {}'.format(filename)
		
		image = cv2.imread(filename)

		if image.shape[0] == 0 or image == None:
			print 'invalid file'
			continue

		target_queue = []
		target_info_queue = []

		target_result_history = np.array([[0,0]])
		target_info_history = []

		height, width = image.shape[:2]

		# Region for sliding windows
		height_search_start = 0
		width_search_start = int(width / 4)

		max_conf = -10.0
		detected_face = None

		for size in DETECT_WINDOW_SIZE:
			height_search_end = height / 3 - size
			width_search_end = width * 3 / 4 - size
			for j in xrange(height_search_start, height_search_end, 3):
				for i in xrange(width_search_start, width_search_end, 3):
					
					target_ = image[j:j+size, i:i+size]

					target = skimage.img_as_float(target_).astype(np.float32)
					target_queue.append(target)
					target_info_queue.append([i, j, size])
					target_info_history.append([i, j, size])

					# prediction = net.predict([target], oversample=False)

					# if prediction[0][1] > max_conf:
					# 	detected_face = {"x":i, "y":j, "size":size}
					# 	max_conf = prediction[0][1]

					if len(target_queue) == 1000:
						prediction = net.predict(target_queue, oversample=False)
						# print type(prediction), prediction[:,1].max()
						local_max = prediction[:,1].max()
						local_max_idx = prediction[:,1].argmax()

						if local_max > max_conf:
							mi, mj, msize = target_info_queue[local_max_idx]
							detected_face = {"x":mi, "y":mj, "size":msize, "conf":local_max}
							max_conf = local_max


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


		
		image2 = image + 0

		target_result_history = target_result_history[1:, 1]
		rank = np.argsort(target_result_history)

		target_info_history = np.array(target_info_history)


		# Green indicates MAX i, j, size
		min_i = 10000
		min_j = 10000
		max_size = 0
		for i in xrange(1, 6):
			info = target_info_history[rank[-i]]
			# print info
			if info[0] < min_i:
				min_i = info[0]
			if info[1] < min_j:
				min_j = info[1]
			if info[2] > max_size:
				max_size = info[2]
			
		# print (min_i, min_j), (min_i+max_size, min_j+max_size)
		cv2.rectangle(image2, (min_i, min_j), (min_i+max_size, min_j+max_size), (0, 255, 0))	

		# # Red indicate AVG
		# mean = np.array([0.0, 0.0, 0.0])
		# for i in xrange(1, 6):
		# 	mean = mean + np.array(target_info_history[rank[-i]]) / 5.0
			
		# mean = mean.astype(np.int64)

		# # print (mean[0], mean[1]), (mean[0]+mean[2], mean[1]+mean[2])
		# cv2.rectangle(image2, (mean[0], mean[1]), (mean[0]+mean[2], mean[1]+mean[2]), (0, 0, 255))			

		# # Blue indicate MAX
		# # print (detected_face['x'], detected_face['y']), (detected_face['x']+detected_face['size'], detected_face['y']+detected_face['size'])
		# cv2.rectangle(image2, (detected_face['x'], detected_face['y']), (detected_face['x']+detected_face['size'], detected_face['y']+detected_face['size']), (255, 0, 0))

		image2 = cv2.resize(image2, (100, 200))
		cv2.imwrite('{}r/test_{}_{}.jpg'.format(image_dir, image_dir, os.path.splitext(os.path.basename(filename))[0]), image2)
		results.append(cv2.resize(image2, (50, 120)))

		time2 = time.clock()

		print "Now computing time : {} sec".format(str(int(time2-time1)))
		

	rs = cv2.hconcat(results)

	cv2.imwrite("result_{}.jpg".format(image_dir), rs)












if __name__ == '__main__':
	main()