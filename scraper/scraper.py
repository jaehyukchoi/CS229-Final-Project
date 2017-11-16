import requests
import time
import codecs
import ast
import os
import cv2
import errno
from bs4 import BeautifulSoup
    
class scraper(object):
    
    def __init__(self, media_file="pages_final.txt"):
        self.acceptable_filetypes = ["jpg","jpeg","png"]
        self.media_file = media_file
        self.session = requests.Session()
        self.login()
        
        self.total_media = 0
        self.num_prev_scraped = 0
        self.num_scraped = 0 
        self.media_info = []
        self.load_media_info()
        self.consecutive_errors = 0
        self.base_url = "http://www.findartinfo.com"
        
    def load_media_info(self):
        
        print("Loading media info. This takes a while...")
        prev_scraped = []
        if("media_metadata.txt" in os.listdir("./")):
            print("Loading metadata file.")
            with codecs.open("media_metadata.txt", "r", "utf-8") as f:
                for line in f:
                    if(line != ""):
                        prev_scraped.append(ast.literal_eval(line)["link"])
        
        self.media_info = []
        with codecs.open(self.media_file, "r", "utf-8") as f:
            for idx,line in enumerate(f):
                if(idx % 10000 == 0):
                    print("Reading line " + str(idx) + "...")
                if(line != ""):
                    self.total_media += 1
                    l_dict = ast.literal_eval(line)
                    if(l_dict["link"] not in prev_scraped):
                        self.media_info.append(l_dict)
                        
        self.num_prev_scraped = len(prev_scraped)
        
    def login(self):

        payload = {
            'Email': 'orbancedric94@gmail.com',
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
            
    def scrape(self):
        
        dst = "./images/"
    
        try:
            os.makedirs(dst)
        except OSError as err:
            if err.errno == errno.EEXIST and os.path.isdir(dst):
                pass
            else:
                raise

        for idx,media in enumerate(self.media_info):
            
            print("[" + str(self.num_prev_scraped+idx+1) + "/" + str(self.total_media) + "] Downloading " + media["title"] + " by " + media["artist"] )
        
            link = media["link"]
            
            timeout = 1
            while True:
                try:
                    r = self.session.get(self.base_url + link)
                    break
                except requests.exceptions.RequestException:
                    print("Connection error, sleeping for " + str(timeout) + " seconds...")
                    for i in range(timeout):
                        time.sleep(1)
                    
                    if(timeout < 64):
                        timeout = timeout*2
            
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
                
                try:
                    dict_key, value = text.split("\n")[0:2]
                except:
                    continue
                
                dict_key = dict_key.lower()
                value = value.lower()
                
                if("signed" in dict_key or "dating" in dict_key):
                    media[dict_key] = value
            
            timeout = 1
            while True:
                try:
                    content = self.session.get(img_src).content
                    break
                except requests.exceptions.RequestException:
                    print("Connection error, sleeping for " + str(timeout) + " seconds...")
                    for i in range(timeout):
                        time.sleep(1)
                        
                    if(timeout < 64):
                        timeout = timeout*2
                        
            image_filename = str(media["sell_price_adjusted"]) + "_" + img_src.split("/")[-1]
            media["filename"] = image_filename
            
            if(image_filename in os.listdir("./images/")):
                continue
            
            try:
                with open(dst + image_filename, 'wb', 0) as media_file:
                    media_file.write(content)
            except Exception as e:
                self.consecutive_errors += 1
                if(self.consecutive_errors > 9):
                    ans = input("WARNING: The last " + str(self.consecutive_errors) + " images have "\
                          "not been successfully downloaded. \nThe last raised exception was: "\
                          + str(e) + "\nPress any key except q to continue.")
                    if("q" in ans):
                        raise
                continue
                
            self.consecutive_errors = 0
            media["state"] = self.resize(image_filename, 400)
                    
            with codecs.open("media_metadata.txt", "a", "utf-8") as f:
                f.write(str(media) + "\n")
                
            self.num_scraped += 1
            
    def resize(self, file, size):
        filetype = file.split(".")[-1]
        
        if(filetype not in self.acceptable_filetypes):
            os.remove("./images/" + file)
            return "Filetype is not compatible."
        
        if os.path.isfile("./images/" + file):
            im = cv2.imread("./images/" + file,-1)
            if im is not None:
                resized_image = cv2.resize(im, (size, size), interpolation = cv2.INTER_CUBIC) 
                cv2.imwrite("./images/" + file, resized_image)
                return "Resized successfully."
            else:
                os.remove("./images/" + file)
                return "Image not loadable."
        else:
            return "Image path not accessible."
                

if __name__ == "__main__":
    
    media_file = "pages_final.txt"
    
    s = scraper(media_file=media_file)
    
    start_time = time.time()
    s.scrape()
    elapsed_time = time.time() - start_time
    
    print(elapsed_time/s.num_scraped)
    