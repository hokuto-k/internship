import cv2
import sys
import csv
import os

# cv2.cv.CV_CAP_PROP_FRAME_WIDTH
cap = cv2.VideoCapture('TownCentreXVID.avi')
dst_dir = sys.argv[1]




while cap.isOpened():
	# Capture frame-by-frame
	ret, frame = cap.read()
	# print a, cap.get(4), cap.get(3)
	# print frame.shape

	curpos = int(cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))

	cv2.imwrite(dst_dir+"/"+str(curpos).zfill(4)+".jpg", frame)
	
	# cv2.imshow('frame',frame)
	if curpos > 1000:
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

