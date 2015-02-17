import cv2
import sys
import csv
import os

# cv2.cv.CV_CAP_PROP_FRAME_WIDTH
cap = cv2.VideoCapture('TownCentreXVID.avi')

top_file = sys.argv[1]
top = open(top_file, 'rb')
reader = csv.reader(top)

top_csv = []

for row in reader:
	top_csv.append(row)

top_csv_by_frame = [[] for i in xrange(4500)]

for row in top_csv:
	top_csv_by_frame[int(row[1])-1].append(row)


target_max = int(sys.argv[2])



while cap.isOpened():
	# Capture frame-by-frame
	ret, frame = cap.read()
	# print a, cap.get(4), cap.get(3)
	# print frame.shape

	curpos = int(cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
	print curpos

	nowframe = curpos - 1
	for target in top_csv_by_frame[nowframe]:
		if int(target[0]) > target_max:
			continue

		dst_dir = target[0]
		if os.path.isdir(dst_dir) == False:
			os.mkdir(dst_dir)

		dst = frame[int(float(target[9])):int(float(target[11])), int(float(target[8])):int(float(target[10]))]
		cv2.imwrite(dst_dir+"/"+str(curpos).zfill(4)+".jpg", dst)
	
	# cv2.imshow('frame',frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

