import os
import sys
import python_module.dpm_cls as dpm_cls
import subprocess
from optparse import OptionParser

usage = "usage: python HeadAngleEstimationFromImage.py [options]"
parser = OptionParser(usage)
parser.add_option("-i", action="store", dest="input_image", help="Input image.")
parser.add_option("-m", action="store", dest="middle_image", help="Output middle image (only DPM).")
parser.add_option("-o", action="store", dest="output_image", help="Output image.")
parser.add_option("-O", action="store", dest="output_text", help="Text file for DPM to CNN.")
parser.add_option("-d", action="store", dest="working_dir", help="Working directory.")
parser.add_option("--dpm_model", action="store", dest="dpm_model", help="Model file for DPM.")
parser.add_option("--cnn_model", action="store", dest="cnn_model", help="Model file for CNN.")
parser.add_option("--cnn_mean", action="store", dest="cnn_mean", help="Mean definition file for CNN.")
parser.add_option("--cnn_deploy", action="store", dest="cnn_deploy", help="Net difinition for deploy file for CNN.")

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

if options.dpm_model:
	dpm_model = options.dpm_model
else:
	dpm_model = "dpm_model/6parts_model"

if options.cnn_model:
	cnn_model = options.cnn_model
else:
	cnn_model = "cnn/model"

if options.cnn_mean:
	cnn_mean = options.cnn_mean
else:
	cnn_mean = "cnn/mean.npy"

if options.cnn_deploy:
	cnn_deploy = options.cnn_deploy
else:
	cnn_deploy = "cnn/deploy"


print options


# do DPM
envs = dict(os.environ)
subprocess.check_call(["./module/dpmdetect_byimage", input_image, middle_image, output_text, dpm_model], env=envs)

# do CNN
dpm_cls.classify_byimage(output_text, input_image, output_image, cnn_model, cnn_deploy, cnn_mean)



