import numpy as np
from sklearn import decomposition

wineData = np.load("./wineArrays/wineData.npy")
wineLabels = np.load("./wineArrays/wineLabels.npy")

pca = decomposition.PCA(11)
pca.fit(wineData)

#3 important
#print(pca.explained_variance_ratio_)

#shuffle data
idx = np.argsort(np.random.random(wineLabels.shape[0]))
wineData = wineData[idx]
wineLabels = wineLabels[idx]

#decimal percent of data that should be used for testing instead of training
split = 0.05

split = round(split * wineLabels.shape[0])
if split < 30:
    print("split less than 30, defaulting to 30")
    split = 30
testData = wineData[:split]
testLabels = wineLabels[:split]
trainData = wineData[split:]
trainLabels = wineLabels[split:]

np.save("./wineArrays/testData.npy", testData)
np.save("./wineArrays/testLabels.npy", testLabels)
np.save("./wineArrays/trainData.npy", trainData)
np.save("./wineArrays/trainLabels.npy", trainLabels)