import requests
import time
import codecs
import ast
import os
import errno
from bs4 import BeautifulSoup
files = os.listdir("./pages/")

def load_media_info():

    media_info = []
    
    for file in files[0:1]:
        print("Reading " + file)
        with codecs.open("./pages/" + file, "r", "utf-8") as f:
            lines = f.read().split("\n")
            lines = [x for x in lines if x != ""]
            
            for l in lines:
                media_info.append(ast.literal_eval(l))
            
    print(len(media_info))
    
    return media_info

def login(session):

    payload = {
        'Email': '',
        'PassWord': '',
        "submitLoginPaid": "Login"
    }
    
    
    r = session.post("http://www.findartinfo.com/login.html")
    soup = BeautifulSoup(r.text, 'html.parser')
    tokens = soup.find_all('input')
    
    for t in tokens:
        try:
            if(t["name"] == "__RequestVerificationToken"):
                payload["__RequestVerificationToken"] = t["value"]
        except KeyError:
            pass
        
    session.post("http://www.findartinfo.com/login.html",data=payload)
    
if __name__ == "__main__":
    
    session = requests.Session()
    media_info = load_media_info()
    login(session)
    
    base_url = "http://www.findartinfo.com"

    media = media_info[0]
    media_page = media["link"]
    r = session.get(base_url + media_page)
    soup = BeautifulSoup(r.text, "html.parser")
    
    images = soup.find_all("img")
    
    img_src = ""
    for img in images:
        if("/Content/Images" not in img["src"]):
            img_src = img["src"] 
            media["src"] = img_src
            
    table = soup.find(id="table6")
    
    dict_key = ""
    store = False
    for tag in table.find_all("tr"):
        text = tag.text.strip()
        dict_key, value = text.split("\n")[0:2]
        
        dict_key = dict_key.lower()
        value = value.lower()
        
        if("signed" in dict_key or "dating" in dict_key):
            media[dict_key] = value
        
        #print(tag.text.strip())
        #print("\n\n")
        
    print(media)
