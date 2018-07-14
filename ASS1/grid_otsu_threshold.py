import sys
import cv2
import argparse
import math
import numpy as np

## set command line arguments ##
parser = argparse.ArgumentParser()
parser.add_argument("--input", help = "You need to input a greyscale image.", required = True)
parser.add_argument("--size", help = "You need to input the size of cells to split.", required = True)
parser.add_argument("--output", help = "You need to input the name of your output image.", required = True)

args = parser.parse_args()

## get args ##
image_org = cv2.imread(args.input,0)
size = int(args.size)
output_name = args.output

## the otsu algrithem ##
def otsu(image_org):
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
	## otsu ##
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
	for i in range(0,h):
		for j in range(0,w):
			if image_org[i][j] <= t_max:
				image_binary[i][j] = 0
			else:
				image_binary[i][j] = 255
	return image_binary, t_max

## jusitify bi-modal histogram ##
def hist_image(image_split, high, low):
	h, w = image_split.shape
	hh = 0
	ll = 255
	for i in range(0, h):
		s = min(image_split[i])
		l = max(image_split[i])
		if l > hh:
			hh = l
		if s < ll:
			ll = s
	if hh < low + 60:
		return 1
	if ll > high - 60:
		return 2
	if hh - ll <= 60:
		return 3
	return 0

## split into cells ##
h, w = image_org.shape
high = 0
low = 255
for i in range(0, h):
	s = min(image_org[i])
	l = max(image_org[i])
	if l > high:
		high = l
	if s < low:
		low = s
image_binary = np.zeros((h,w), np.uint8)
t_list = np.zeros((h,w), np.uint8)
for x in range(0, h, size):
	for y in range(0, w, size):
		image_split = image_org[x:x+size, y:y+size]
		## justify whether a bi-modal histogram ##
		result = hist_image(image_split, high, low)
		if result == 0:
			## apply otsu algrithem ##
			image_binary[x:x+size, y:y+size] ,t= otsu(image_split)
			t_list[x:x+size, y:y+size] += t
		else:
			## write a bianry image based on otsu threshold ##
			if len(t_list) == 0:
				if result == 2:
					image_binary[x:x+size, y:y+size] += 255
				if result == 3:
					hh = 0
					ll = 255
					for i in range(0, len(image_split)):
						s = min(image_split[i])
						l = max(image_split[i])
						if l > hh:
							hh = l
						if s < ll:
							ll = s
					if (int(ll)+int(hh)) > (int(low)+int(high)):
						image_binary[x:x+size, y:y+size] += 255		
			else:
				if y == 0:
					t_now = t_list[x-1][y]
				if x == 0:
					t_now = t_list[x][y-1]
				if x!=0 and y!=0:
					t_now = t_list[x][y-1]
				for i in range(0,len(image_split)):
					for j in range(0,len(image_split[i])):
						if image_split[i][j] <= t_now:
							image_binary[x+i][y+j] = 0
						else:
							image_binary[x+i][y+j] = 255
				t_list[x:x+size, y:y+size] += t_now
cv2.imwrite(output_name, image_binary)
