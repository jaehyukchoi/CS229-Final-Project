import ast
import codecs
from easymoney.money import EasyPeasy

def adjust_for_inflation(media_info):
    
    ep = EasyPeasy()
    
    media_info_adjusted = []

    for idx,m in enumerate(media_info):
        
        if(idx%10000 == 0):
            print("Adjusting price [" + str(idx) + "/" + str(len(media_info)) + "]")

        sell_date_year = m["sell_date"].split("-")[-1]
        sell_price = m["sell_price"]
        sell_price_adjusted = ep.normalize(amount=int(sell_price), region="US", \
                                           from_year=int(sell_date_year), to_year="latest",base_currency = "USD")
        sell_price_adjusted = int(round(sell_price_adjusted))
        m["sell_price_adjusted"] = sell_price_adjusted
        media_info_adjusted.append(m)
        #print(sell_date_year + ":" + sell_price + " -> " + str(sell_price_adjusted))
    
    with codecs.open("./pages_final.txt", "w", "utf-8") as f:
        for m in media_info_adjusted:
            f.write(str(m) + "\n")
            
if __name__ == "__main__":
    
    media_info = []
    
    print("Reading pages_unique.txt")
    with codecs.open("./pages_unique.txt", "r", "utf-8") as f:
        lines = f.read().split("\n")
        lines = [x for x in lines if x != ""]
        for idx,l in enumerate(lines):
            
            if(idx % 10000 == 0):
                print("Reading in line " + str(idx) + "/" + str(len(lines)))
            
            media_info.append(ast.literal_eval(l))
            
            
    adjust_for_inflation(media_info)