import os
import Image
import sys


count = 0
for root, dirs, files in os.walk(sys.argv[1]):
	for file in files:
		filename = os.path.join(root, file)
		body, ext = os.path.splitext(file)
		if ext == '.png' or ext == '.jpg':
			img = Image.open(filename)
			img.crop((15, 15, 58, 121)).save('{}/{}.jpg'.format(sys.argv[2], str(count).zfill(5)), 'JPEG')
			count += 1
			# except:
			# 	print filename