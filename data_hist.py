import ast
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

def histedges_equalN(x, nbin):
	npt = len(x)
	return np.interp(np.linspace(0, npt, nbin + 1),
					 np.arange(npt),
					 np.sort(x))

if __name__ == "__main__":
	
	media_file = "pages_final_vidush.txt"
	# with open(media_file,'r', encoding="utf8") as file:
	#   i = 1
	#   sell_prices = []
	#   for line in file:
	#       print(i)
	#       data_dict = ast.literal_eval(line)
	#       sell_prices.append(int(data_dict['sell_price_adjusted']))
	#       i += 1
	#   sell_prices = np.array(sell_prices)
	#   np.save('sell_prices',sell_prices)
	sell_prices = np.load('sell_prices.npy')
	print(sell_prices)
	n, bins, patches = plt.hist(sell_prices, histedges_equalN(sell_prices, 10))
	print(bins)
	plt.hist(sell_prices, histedges_equalN(sell_prices, 10))
	plt.show()

	# artist_num_of_works = {}
	# with open(media_file,'r', encoding="utf8") as file:
	#   i = 1
	#   for line in file:
	#       print(i)
	#       data_dict = ast.literal_eval(line)
	#       artist_name = data_dict['artist']
	#       artist = ''
	#       for c in artist_name:
	#           if c == '(':
	#               break
	#           else:
	#               artist += c
	#       artist = artist[:-1]
	#       print(artist)
	#       i += 1
	#       if artist in artist_num_of_works.keys():
	#         artist_num_of_works[artist] += 1
	#       else:
	#         artist_num_of_works[artist] = 1
	#   with open('artist_dict.pkl', 'wb') as pickle_file:
	#       pickle.dump(artist_num_of_works, pickle_file, protocol=4)
	# with open('artist_dict.pkl', 'rb') as pickle_load:
	# 	artist_num_of_works = pickle.load(pickle_load)
	# 	artist_num_of_works.pop('')
	# 	# artist_num_of_works.pop('William')
	# 	max_works_artist, value = max(artist_num_of_works.items(), key = lambda p: p[1])
	# 	print((max_works_artist,value))
		

		
