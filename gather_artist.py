import requests
import time
import codecs
from bs4 import BeautifulSoup

base_url = "http://www.findartinfo.com/"
search = "english/Artists/Result?artistName=&born=&die=&countryId="


session = requests.Session()

pages = 14601
info = []

start_time = time.time()

for i in range(pages):
    
    r = session.get(base_url + search + "&pageIndex=" + str(i+1))

    soup = BeautifulSoup(r.text, "html.parser")
        
    for tag in soup.find_all("tr"):
        if(tag.has_attr("onmouseout")):
            
            artist_info = {}
            
            for subtag in tag.find_all("a"):
                
                try:
                    if("Last Hammer Price" in subtag["title"]):
                        artist_info["name"] = subtag.text.strip()
                        artist_info["link"] = subtag["href"]
                except KeyError:
                    if("list-prices-by-artist" in subtag["href"]):
                        artist_info["prices"] = subtag.text.strip()
                
                if("list-prices-has-art-work-by-artist" in subtag["href"]):
                        artist_info["pictures"] = subtag.text.strip()
    
        
            if(int(artist_info["prices"]) != 0 and int(artist_info["pictures"]) != 0):
                print(str(artist_info["name"]) + " - " + str(artist_info["prices"]) + " - " + str(artist_info["pictures"]))
                info.append(artist_info)
            
print("Time per page = " + str((time.time() - start_time)/pages)[0:4])

print(str(len(info)) + " links gathered.")

with codecs.open("links.txt", "w", "utf-8") as f:
    
    for i in info:
        f.write(str(i) + "\n")
    




