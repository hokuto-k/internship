import Image
import sys
import os
import random

src_dir = sys.argv[1]
dst_dir = sys.argv[2]
_width = int(sys.argv[3])
_height = int(sys.argv[4])

count = 0

for root, dirs, files in os.walk(src_dir):
	for file_ in files:
		filename = os.path.join(root, file_)
		img = Image.open(filename)
		width, height = img.size 

		for i in xrange(2):
			x = random.randint(0, width-_width)
			y = random.randint(0, height-_height)
			box = (x, y, x+_width, y+_height)
			dst = img.crop(box)
			dst.save('{}/{}.jpg'.format(dst_dir, str(count).zfill(5)))
			count += 1


