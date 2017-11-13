import requests
import time
import codecs
import ast
import os
import errno
from bs4 import BeautifulSoup

def load_artist_info():

    artist_info = []
    
    with codecs.open("links.txt", "r", "utf-8") as f:
        lines = f.read().split("\n")
        lines = [x for x in lines if x != ""]
        
        for l in lines:
            artist_info.append(ast.literal_eval(l))
                
    return artist_info
    
def gather_media_pages(artist_info, rnd, filename):
    
    info = []
    session = requests.Session()

    base_url = "http://www.findartinfo.com"
    
    for idx,artist in enumerate(artist_info):
        
        artist_page = artist["link"]
        
        print("Batch " + str(rnd) + " [" + str(idx+1) + "/" + str(len(artist_info)) + "] Scraping " + artist["name"])
            
        num_pages = 1
        i = 0
        
        while i < num_pages:
            
            i += 1
            
            print("Page " + str(i) + "...")
            
            r = session.get(base_url + artist_page + str(i) + ".html")
            soup = BeautifulSoup(r.text, "html.parser")
            
            if(i == 1):
                header = soup.find_all("h2")[0]
                header_page_info = header.text.split("|")[-2]
                of_idx = header_page_info.find("of")
                par_idx = header_page_info.find("(")
                num_pages = int(header_page_info[of_idx + 3: par_idx])
        
            for tag in soup.find_all("tr"):
                if(tag.has_attr("onmouseout")):
                    piece_info = {}
                    for idx,subtag in enumerate(tag.find_all("td")):
                        
                        if(idx == 0 and len(subtag.find_all("img")) == 1):
                            break     
                        elif(idx == 1):
                            piece_info["sell_date"] = subtag.text.strip()
                        elif(idx == 2):
                            piece_info["title"] = subtag.text.strip()
                            piece_info["link"] = subtag.find_all("a")[0]["href"]
                        elif(idx == 3):
                            piece_info["dimensions"] = subtag.text.strip()
                        elif(idx == 4):
                            piece_info["medium"] = subtag.text.strip()
                        elif(idx == 5):
                            piece_info["sell_price"] = subtag.text.strip()
                        
                    if(len(piece_info) != 0 and "Unsold" not in piece_info["sell_price"]):
                        piece_info["artist"] = artist["name"]
                        info.append(piece_info)
                                            
    with codecs.open(filename, "w", "utf-8") as f:
        for i in info:
            f.write(str(i) + "\n")
            
    return info

if __name__ == "__main__":
    
    print("Reading links.txt...")
    artist_info = load_artist_info()[0:1000]
    
    dst = "./pages/"
    
    try:
        os.makedirs(dst)
    except OSError as err:
        if err.errno == errno.EEXIST and os.path.isdir(dst):
            pass
        else:
            raise
    
    batch_size = 200
    
    batches = len(artist_info)//batch_size
    left_over = len(artist_info) % batch_size
    total = 0
    total_start_time = time.time()
    
    for i in range(batches):
        filename = "media_pages_" + str(i) + ".txt"
        start_time = time.time()
        info = gather_media_pages(artist_info[batch_size*(i) :batch_size*(i) + batch_size], i+1, dst + filename)
        print("Gathered " + str(len(info)) + " media pages in this batch.")
        total += len(info)
        elapsed_time = time.time()- start_time
        print("Average = " + str(elapsed_time/100))
        
    info = gather_media_pages(artist_info[batch_size*batches :batch_size*batches + left_over - 1], batches+1, dst + filename)
    total += len(info)
    total_elapsed_time = time.time() - start_time
    
    print("Total Elapsed Time = " + str(elapsed_time))
    print("Gathered " + str(total) + " total media pages.")
    print("Time per media page = " + str(elapsed_time/total))
