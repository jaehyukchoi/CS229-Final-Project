#!/usr/bin/python
from PIL import Image
import os, sys


cwd = os.getcwd()
im_path = cwd + "/images/"
resize_path = cwd + "/resized images/"
im_dirs = os.listdir( im_path )

def resize(size):
    for item in im_dirs:
        if os.path.isfile(im_path+item):
            im = Image.open(im_path+item)
            imResize = im.resize((size,size), Image.ANTIALIAS)
            imResize.save(resize_path + item[:-4] + '_resized.jpg', 'JPEG', quality=90)

size = input('What size do you want the images?')
resize(int(size))