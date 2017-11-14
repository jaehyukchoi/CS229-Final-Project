import ast
import codecs
import os

def load_media_info():

    files = os.listdir("./pages/")
    
    media_info = []
    
    for file in files[-1:]:
        print("Reading " + file)
        with codecs.open("./pages/" + file, "r", "utf-8") as f:
            lines = f.read().split("\n")
            lines = [x for x in lines if x != ""]
            
            for l in lines:
                media_info.append(ast.literal_eval(l))
            
    #print(len(media_info))
    
    return media_info

def clean_pricing(media_info):
    cnt = 0
    num_media = len(media_info)

    for idx,media in enumerate(media_info):
        
        #print("[" + str(idx+1) + "/" + str(num_media) + "] Cleaning " + media["title"] + " by " + media["artist"] )

        price = media["sell_price"].split(" ")[0].replace(",","")
        media["sell_price"] = price
        
        print(price)
        
    
            
if __name__ == "__main__":
    m = load_media_info()
    clean_pricing(m)
    