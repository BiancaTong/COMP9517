import sys
import cv2
import argparse
import math
import numpy as np

## set command line arguments ##
parser = argparse.ArgumentParser()
parser.add_argument("--input", help = "You need to input a greyscale image.", required = True)
parser.add_argument("--output", help = "You need to input the name of your output image.", required = True)
parser.add_argument("--threshold", action = "store_true", help = "You need to decide whether print the threshold found.", required = False)
args = parser.parse_args()

## get args ##
image_org = cv2.imread(args.input,0)
output_name = args.output
print_flag = args.threshold

## compute the range of the image ##
h, w = image_org.shape
low = 255
high = 0
for x in range(0, h):
	s = min(image_org[x])
	l = max(image_org[x])
	if l > high:
		high = l
	if s < low:
		low = s

## the otsu algrithem ##
bcv_max = 0
t_max = 0
for t in range(low, high+1):
	## Separate the pixels into two clusters according to the threshold ##
	fore_ground = (image_org <= t)*image_org
	back_ground = (image_org > t)*image_org
	f_sum = fore_ground.sum()
	b_sum = back_ground.sum()
	f_num = len(fore_ground[fore_ground.nonzero()])
	b_num = len(back_ground[back_ground.nonzero()])
	if f_num == 0:
		break
	if b_num > 0:
		## Find the mean of each cluster ##
		mean_fore = f_sum*1.0/f_num
		mean_back = b_sum*1.0/b_num
		## Square the difference between the means ##
		mean_square = math.pow(abs(mean_fore - mean_back),2)
		## Multiply the number of pixels in one cluster by the number in the other ##
		between_class_variance = f_num * b_num * mean_square * 1.0
		## Find the largest value ##
		if between_class_variance > bcv_max:
			bcv_max = between_class_variance
			t_max = t

## write a bianry image based on otsu threshold ##
image_binary = np.zeros((h,w), np.uint8)
for i in range(0, h):
	for j in range(0, w):
		if image_org[i][j] <= t_max:
			image_binary[i][j] = 0
		else:
			image_binary[i][j] = 255
cv2.imwrite(output_name, image_binary)

## print the threshold found ##
if print_flag is True:
	print("the ostu threshold is {}".format(t_max))
