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
count = 0
all_count = 0


while cap.isOpened():
	# Capture frame-by-frame
	ret, frame = cap.read()
	width = frame.shape[1]
	height = frame.shape[0]
	# print width, height
	# print a, cap.get(4), cap.get(3)
	# print frame.shape

	curpos = int(cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
	# print curpos

	nowframe = curpos - 1
	if nowframe >= 4499:
		break

	if nowframe % 30 != 0:
		continue

	for target in top_csv_by_frame[nowframe]:
		all_count += 1
		if int(target[0]) > target_max:
			continue

		dst_dir = "TC_heads/" + target[0].zfill(3)
		if os.path.isdir(dst_dir) == False:
			os.mkdir(dst_dir)

		hx = int(float(target[4])) - 5
		hy = int(float(target[5])) - 5
		hx2 = int(float(target[6])) + 5
		hy2 = int(float(target[7])) + 5
		if hx2 - hx < 40 or hx < 0 or hy < 0 or hx2 > width or hy2 > height:
			continue
		else:
			count += 1

		dst = frame[hy:hy2, hx:hx2]
		# cv2.imwrite(dst_dir+"/"+str(curpos).zfill(4)+".jpg", dst)
		cv2.imwrite(str(count).zfill(4)+".jpg", dst)
	
		if count % 100 == 0:
			print count, "/", all_count 
	# cv2.imshow('frame',frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

