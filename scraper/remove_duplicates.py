import ast
import codecs
import os
import errno
import time
from  more_itertools import unique_everseen


def load_media_info():
    
    src = "./pages_clean/"

    files = os.listdir(src)

    files = sorted(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    
    media_info = []
    
    for file in files:
        print("Reading " + file)
        with codecs.open(src + file, "r", "utf-8") as f:
            lines = f.read().split("\n")
            lines = [x for x in lines if x != ""]
            
            for l in lines:
                media_info.append(ast.literal_eval(l))
                
    return media_info

def remove_duplicates(media_info):
    print("Removing duplicates...")
    
    #start_time = time.time()
    #u = list(unique_everseen(media_info))
    #print(time.time()-start_time)

    start_time = time.time()
    u = [dict(t) for t in set([tuple(d.items()) for d in media_info])]
    print(time.time()-start_time)

    #print(len(u))
    #print(len(u)/len(media_info))
    
    with codecs.open("pages_unique.txt", "w", "utf-8") as f:
        for media in u:
            f.write(str(media) + "\n")
        
if __name__ == "__main__":
    
    m = load_media_info()
    remove_duplicates(m)