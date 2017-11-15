import requests
import time
import codecs
import ast
import os
import errno
from bs4 import BeautifulSoup

cedric = []
vid = []
rafi = []

with codecs.open("pages_final.txt", "r", "utf-8") as f:
    lines = f.read().split("\n")
    lines = [x for x in lines if x != ""]
    boundary = len(lines)//3
    
    for l in lines[0:boundary]:
        cedric.append(ast.literal_eval(l))
        
    for l in lines[boundary:boundary*2]:
        vid.append(ast.literal_eval(l))
        
    for l in lines[boundary*2:]:
        rafi.append(ast.literal_eval(l))
        
print(len(cedric))
print(len(vid))
print(len(rafi))

with codecs.open("pages_final_cedric.txt", "w", "utf-8") as f:
    for l in cedric:
        f.write(str(l) + "\n")
        
with codecs.open("pages_final_vidush.txt", "w", "utf-8") as f:
    for l in vid:
        f.write(str(l) + "\n")
        
with codecs.open("pages_final_rafi.txt", "w", "utf-8") as f:
    for l in rafi:
        f.write(str(l) + "\n")
    