#!/usr/bin/python
import cv2
import os, sys


cwd = os.getcwd()
im_path = cwd + "/images/"
im_dirs = os.listdir( im_path )

def resize(size):
<<<<<<< HEAD
<<<<<<< HEAD
    if os.path.isfile(im_path+item):
        im = cv2.imread(im_path+item,-1)
        if im is not None:
        	resized_image = cv2.resize(im, (size, size), interpolation = cv2.INTER_CUBIC) 
        	cv2.imwrite(resize_path + item, resized_image)
        else:
        	continue
=======
=======
>>>>>>> ec9e43fe5c09ebd5c2bc91e0886a97cdb3c25067
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
<<<<<<< HEAD
>>>>>>> ec9e43fe5c09ebd5c2bc91e0886a97cdb3c25067
=======
>>>>>>> ec9e43fe5c09ebd5c2bc91e0886a97cdb3c25067

size = input('What size do you want the images?')
resize(int(size))