#!/usr/bin/python
import cv2
import os, sys


cwd = os.getcwd()
im_path = cwd + "/images/"
im_dirs = os.listdir( im_path )

def resize(size):
	for item in im_dirs:
		if os.path.isfile(im_path+item):
			im = cv2.imread(im_path+item,-1)
			if im is not None and item[-3:] == 'jpg':
				resized_image = cv2.resize(im, (size, size), interpolation = cv2.INTER_CUBIC) 
				cv2.imwrite(im_path + item, resized_image)
				print('Writing Resized: ' + item)
			else:
				os.remove(im_path+item)
				continue
		else:
			try: 
				os.remove(im_path+item)
			except: 
				pass

size = input('What size do you want the images?')
resize(int(size))