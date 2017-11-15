import ast
import codecs
import os
import errno

def clean_pricing(dst):

    files = os.listdir("./pages/")
    files = sorted(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    
    for file in files:
        file_number = int(file.split("_")[-1].split(".")[0])
        media_cleaned = []
        print("Cleaning " + file)
        with codecs.open("./pages/" + file, "r", "utf-8") as f:
            lines = f.read().split("\n")
            lines = [x for x in lines if x != ""]
            
            for l in lines:
                
                media = ast.literal_eval(l)
                
                price = media["sell_price"].split(" ")[0].replace(",","")
                media["sell_price"] = price
                media_cleaned.append(media)
        
        filename = "media_page_clean_" + str(file_number) + ".txt"
        with codecs.open(dst + filename, "w", "utf-8") as f:           
            for m in media_cleaned:
                f.write(str(m) + "\n")
                                    
if __name__ == "__main__":
    
    dst = "./pages_clean/"
    
    try:
        os.makedirs(dst)
    except OSError as err:
        if err.errno == errno.EEXIST and os.path.isdir(dst):
            pass
        else:
            raise
    
    clean_pricing(dst)