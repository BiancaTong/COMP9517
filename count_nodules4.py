import sys
import cv2
import argparse
import math
import numpy as np
import copy

## set command line arguments ##
parser = argparse.ArgumentParser()
parser.add_argument("--input", help = "You need to input a binary image.", required = True)
parser.add_argument("--size", help = "You need to input the size of minimum area.", required = True)
parser.add_argument("--optional_output", help = "You need to input the name of your output image.", required = False)

args = parser.parse_args()

## get args ##
image_org = cv2.imread(args.input,-1)
size = int(args.size)
output_name = args.optional_output

## find the smallest label of neighbours ##
def neigh(label,i,j,set_eq):
	## find all the neighbours ##
	if j == 0 and i == 0:
		neighbour = [label[i+1][j], label[i][j+1]]
	elif j == len(label[i])-1 and i == 0:
		neighbour = [label[i+1][j], label[i][j-1]]
	elif j == 0 and i == len(label)-1:
		neighbour = [label[i-1][j], label[i][j+1]]
	elif j == len(label[i])-1 and i == len(label)-1:
		neighbour = [label[i-1][j], label[i][j-1]]
	elif j == 0 and i != 0 and i != len(label)-1:
		neighbour = [label[i-1][j], label[i+1][j], label[i][j+1]]
	elif j == len(label[i])-1 and i != 0 and i != len(label)-1:
		neighbour = [label[i-1][j], label[i+1][j], label[i][j-1]]
	elif j != 0  and j != len(label[i])-1 and i == 0:
		neighbour = [label[i+1][j], label[i][j+1], label[i][j-1]]
	elif j != 0  and j != len(label[i])-1 and i == len(label)-1:
		neighbour = [label[i-1][j], label[i][j+1], label[i][j-1]]
	elif j != 0 and i != 0 and j != len(label[i])-1 and i != len(label)-1:
		neighbour = [label[i-1][j], label[i+1][j], label[i][j+1], label[i][j-1]]
	## find the smallest label and construct the eq list ##
	nf = filter(lambda a: a != 0, neighbour)
	if len(nf) == 0:
		return None,None
	else:
		for n in nf:
			x = filter(lambda a: a != n, nf)
			for m in x:
				if m not in set_eq[n]:
					set_eq[n].append(m)
				if n not in set_eq[m]:
					set_eq[m].append(n)
				for y in set_eq[n]:
					if m not in set_eq[y]:
						set_eq[y].append(m)
					if y not in set_eq[m]:
						set_eq[m].append(y)
				for k in set_eq[m]:
					if n not in set_eq[k]:
						set_eq[k].append(n)
					if k not in set_eq[n]:
						set_eq[n].append(k)
		return min(nf),set_eq
		
## two pass algorithm ##

## first pass ##
h, w= image_org.shape
label = [[0 for col in range(w)] for row in range(h)]
set_eq = [[]]
new_label = 1
for i in range(0, h):
	for j in range(0, w):
		if image_org[i][j] != 255:
			mini_label, set_eq_update = neigh(label,i,j,set_eq)
			if mini_label is None:
				label[i][j] = new_label
				set_eq.append([new_label])
				new_label += 1
			else:
				label[i][j] = mini_label
				set_eq = copy.deepcopy(set_eq_update)

## second pass ##
for i in range(0, h):
	for j in range(0, w):
		if label[i][j] != 0:
			re_label = min(set_eq[label[i][j]])
			label[i][j] = re_label

## output nodules number ##
n_num = 0
eq_dict = {}
for i in range(0, h):
	for j in range(0, w):
		if label[i][j] not in eq_dict:
			eq_dict.update({label[i][j]:1})
		else:
			eq_dict[label[i][j]]+=1
for i in eq_dict:
	if eq_dict[i] > size and i != 0:
		n_num+=1
print("The number of nodules with area larger than {} pixels in the image: {}".format(size, n_num))

## output image ##
if output_name:
	image_nodules = np.zeros((h,w,3), np.uint8)
	color_dict = {0:[255,255,255]}
	color_list = [[255,255,255]]
	for i in range(0, h):
		for j in range(0, w):
			if eq_dict[label[i][j]] > size:
				if label[i][j] == 0:
					image_nodules[i][j] = [255,255,255]
				else:
					if label[i][j] in color_dict:
						image_nodules[i][j] = color_dict[label[i][j]]
					else:
						color_random = list(np.random.random(size=3) * 256) 
						while color_random in color_list:
							color_random = list(np.random.random(size=3) * 256) 
						color_list.append(color_random)
						color_dict.update({label[i][j]:color_random})
						image_nodules[i][j] = color_dict[label[i][j]]
			else:
				image_nodules[i][j] = [255,255,255]
	cv2.imwrite(output_name, image_nodules)
	cv2.imshow('image_nodules', image_nodules)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

