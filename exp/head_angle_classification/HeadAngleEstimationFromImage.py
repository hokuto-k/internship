import os
import sys
import dpm_cls
from optparse import OptionParser

usage = "usage: python HeadAngleEstimationFromImage.py [-i input_image] [-m middle_image] [-o output_image] [-O output_text] [-d working_dir]"
parser = OptionParser(usage)
parser.add_option("-i", action="store", dest="input_image", help="Input image.")
parser.add_option("-m", action="store", dest="output_image", help="Output middle image (only DPM).")
parser.add_option("-o", action="store", dest="output_image", help="Output image.")
parser.add_option("-O", action="store", dest="output_text", help="Text file for DPM to CNN.")
parser.add_option("-d", action="store", dest="working_dir", help="Working directory.")
(options, args) = parser.parse_args()

if options.working_dir:
	working_dir = options.working_dir
else:
	working_dir = "working"

if not os.path.isdir(working_dir):
	os.mkdir(working_dir)


if options.input_image:
	input_image = options.input_image
else:
	input_image = "sample/tc0001.jpg"

if options.output_image:
	output_image = os.path.join(working_dir, options.output_image)
else:
	output_image = os.path.join(working_dir, "output.jpg")

if options.middle_image:
	middle_image = os.path.join(working_dir, options.middle_image)
else:
	middle_image = os.path.join(working_dir, "middle.jpg")

if options.output_text:
	output_text = os.path.join(working_dir, options.output_text)
else:
	output_text = os.path.join(working_dir, "output.txt")

print options


# do DPM
envs = dict(os.environ)
subprocess.check_call(["./dpmdetect_byimage", input_image, middle_image, output_text], env=envs)

# do CNN
dpm_cls.classify_byimage(output_text, input_image, working_dir, output_image, model, mean, deploy)



