import codecs
import ast
import numpy as np
import operator
from scipy.io import savemat

filename = "media_metadata2.txt"
with codecs.open(filename, "r", "utf-8") as f:
    lines = f.readlines()
    f.close()

alldating = []
allsigned = []
allmedium = []
for ind,dic in enumerate(lines):
    print(ind)
    tempdict = ast.literal_eval(lines[ind])
    if 'signed' in tempdict:
        allsigned.append(tempdict['signed'])
    if 'dating' in tempdict:
        alldating.append(tempdict['dating'])
    if 'medium' in tempdict:
        allmedium.append(tempdict['medium'])

# Go through again and get an approximate count of each class
datingdict = {}
signeddict = {}
mediumdict = {}
artistlist = []
for ind,dic in enumerate(lines):
    print(ind)
    tempdict = ast.literal_eval(lines[ind])
    artistlist.append(tempdict['artist'])
    if 'signed' in tempdict:
        if tempdict['signed'] not in signeddict:
            signeddict[tempdict['signed']] = 0
        signeddict[tempdict['signed']] += 1
    if 'dating' in tempdict:
        if tempdict['dating'] not in datingdict:
            datingdict[tempdict['dating']] = 0
        datingdict[tempdict['dating']] += 1
    if 'medium' in tempdict:
        if tempdict['medium'] not in mediumdict:
            mediumdict[tempdict['medium']] = 0
        mediumdict[tempdict['medium']] += 1

# Sort dict and print highest values
sorted_signed = sorted(signeddict.items(), key=operator.itemgetter(1), reverse=True)
sorted_dating = sorted(datingdict.items(), key=operator.itemgetter(1), reverse=True)
sorted_medium = sorted(mediumdict.items(), key=operator.itemgetter(1), reverse=True)
mediumlist = []
for ind,ist in enumerate(sorted_signed):
    print(ist[0] + ': ' + str(ist[1]))
for ind,ist in enumerate(sorted_dating):
    #if ind > 100: continue
    print(ist[0] + ': ' + str(ist[1]))
for ind,ist in enumerate(sorted_medium):
    #if ind > 100: continue
    if ist[1] < 100: break
    mediumlist.append(ist[0])
    print(ist[0] + ': ' + str(ist[1]))

# Should contain 250 artists
artistlist = list(set(artistlist))
print('# of artists: ' + str(len(artistlist)))
# Should contain about 15 mediums
print('# of mediums: ' + str(len(mediumlist)))

# Create big dict of all features
featlist = ['Signed', 'Dating'] + artistlist + mediumlist
featdict = {el:0 for el in featlist}

# Create an input matrix with every sample
inputmat = []
# All the classes for pricing
classes = [[0,101],[101,196],[196,321],[321,499],[499,760],[760,1202],[1202,2033],[2033,3857],[3857,9643],[9643,60130038]]
# Create class matrix
classmat = []

# Okay now we loop through one last time and create our data. We'll make dicts for each piece
for ind,dic in enumerate(lines):
    print(ind)
    tempdict = ast.literal_eval(lines[ind])
    piecedict = dict(featdict)
    yvec = np.zeros((len(classes),))

    # Try to get dating. If not, skip
    try:
        date = int(tempdict['dating'][-4:])
    except ValueError:
        continue
    piecedict['Dating'] = 2017 - date

    # Get if signed
    if 'signed' in tempdict:
        if tempdict['signed'] == 'yes':
            piecedict['Signed'] = 1
        else:
            piecedict['Signed'] = 0
    else:
        piecedict['Signed'] = 0

    # Get artist
    piecedict[tempdict['artist']] = 1
    # Get medium
    if tempdict['medium'] in mediumlist:
        piecedict[tempdict['medium']] = 1
    else:
        continue

    # Get pricing
    price_label = tempdict['sell_price_adjusted']
    for i in range(len(classes)):
        if price_label >= classes[i][0] and price_label < classes[i][1]:
            yvec[i] = 1

    # Transform dict into list
    piecelist = list(piecedict.values())
    # Append to master list
    inputmat.append(piecelist)
    # Append to class list
    classmat.append(yvec)

# Turn inputmat to array and save as a mat file
savemat('NNdata.mat', mdict={'xarr': inputmat, 'yarr' : classmat})

# Save numpy array to file
np.save('xdata.npy', inputmat)
np.save('ydata.npy', classmat)
