import requests
import time
import codecs
import ast
import os
import errno
from bs4 import BeautifulSoup

def load_media_info():
    
    src = "./pages_clean/"

    files = os.listdir(src)

    files = sorted(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    
    media_info = []
    
    for file in files[0:1]:
        print("Reading " + file)
        with codecs.open(src + file, "r", "utf-8") as f:
            lines = f.read().split("\n")
            lines = [x for x in lines if x != ""]
            
            for l in lines:
                media_info.append(ast.literal_eval(l))
                
    return media_info
    
class scraper(object):
    
    def __init__(self, media_info):
        self.session = requests.Session()
        self.login(self.session)
        
        self.media_info = media_info
        self.base_url = "http://www.findartinfo.com"
        self.num_media = len(self.media_info)
        
    def login(self):

        payload = {
            'Email': '',
            'PassWord': '',
            "submitLoginPaid": "Login"
        }
        
        r = self.session.post("http://www.findartinfo.com/login.html")
        soup = BeautifulSoup(r.text, 'html.parser')
        tokens = soup.find_all('input')
        
        for t in tokens:
            try:
                if(t["name"] == "__RequestVerificationToken"):
                    payload["__RequestVerificationToken"] = t["value"]
            except KeyError:
                pass
            
        self.session.post("http://www.findartinfo.com/login.html",data=payload)
        
    
    def scrape(self, filename):
        
        dst = "./images/"
    
        try:
            os.makedirs(dst)
        except OSError as err:
            if err.errno == errno.EEXIST and os.path.isdir(dst):
                pass
            else:
                raise
        
        for idx,media in enumerate(media_info):
            
            print("[" + str(idx+1) + "/" + str(num_media) + "] Downloading " + media["title"] + " by " + media["artist"] )
        
            link = media["link"]
            r = session.get(base_url + link)
            soup = BeautifulSoup(r.text, "html.parser")
            
            images = soup.find_all("img")
            
            img_src = ""
            for img in images:
                if("/Content/Images" not in img["src"]):
                    img_src = img["src"] 
                    media["src"] = img_src
                    
            table = soup.find(id="table6")
            
            for tag in table.find_all("tr"):
                text = tag.text.strip()
                dict_key, value = text.split("\n")[0:2]
                
                dict_key = dict_key.lower()
                value = value.lower()
                
                if("signed" in dict_key or "dating" in dict_key):
                    media[dict_key] = value
                    
            try:
                content = session.get(img_src).content
            except (requests.exceptions.ConnectionError,requests.exceptions.ConnectionResetError):
                time.sleep(5)
                content = session.get(img_src).content
            
            #price = med
            image_filename = media["sell_price"] + "_" + img_src.split("/")[-1]
            media["filename"] = image_filename
            
            if(image_filename in os.listdir("./images/")):
                continue
    
            with open(dst + image_filename, 'wb') as media_file:
                media_file.write(content)
                    
            with codecs.open(filename, "a", "utf-8") as f:
                f.write(str(media) + "\n")
            
if __name__ == "__main__":
    
    download = 100
    
    media_info = load_media_info()
    start_time = time.time()
    scrape(media_info[0:download], "metadata.txt")
    elapsed_time = time.time() - start_time
    
    print(elapsed_time)
    print(elapsed_time/download)
