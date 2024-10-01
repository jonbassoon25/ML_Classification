import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition

def generateData(pca, trainData):
    original = pca.components_.copy()
    ncomp = pca.components_.shape[0]
    a = pca.transform(trainData)
    for i in range(2, ncomp):
        pca.components_[i,:] += np.random.normal(scale = 0.1, size = ncomp)
    b = pca.inverse_transform(a) #returns the data in it's original format instead of being sorted
    pca.components_ = original.copy()
    return b

labels = np.load("./irisArrays/irisLabels.npy")
data = np.load("./irisArrays/irisData.npy")

pca = decomposition.PCA(n_components=4)
pca.fit(data)

analysis = pca.explained_variance_ratio_

print(analysis)

#shuffle data
idx = np.argsort(np.random.random(labels.shape[0])) #sort random values into indicies
shuffledData = data[idx] #reassign indexes of data
shuffledLabels = labels[idx] #reassign indexes of labels

testLabels = shuffledLabels[:30]
trainLabels = shuffledLabels[30:]
testData = shuffledData[:30]
trainData = shuffledData[30:]

pca.fit(shuffledData)

#create augmented data
augLabels = np.zeros(len(trainLabels) * 10, dtype="uint8")
augData = np.zeros((len(trainLabels) * 10, 4))

for i in range(10):
    if i == 0:
        augLabels[0:trainData.shape[0]] = trainLabels
        augData[0:trainData.shape[0],:] = trainData
    else:
        augLabels[i * trainData.shape[0] : i * trainData.shape[0] + trainData.shape[0]] = trainLabels
        augData[i * trainData.shape[0] : i * trainData.shape[0] + trainData.shape[0],:] = generateData(pca, trainData)



idx = np.argsort(np.random.random(augData.shape[0]))
q = augData[idx]
r = augLabels[idx]
np.save("./irisArrays/augmentedIrisData.npy", q)
np.save("./irisArrays/augmentedIrisLabels.npy", r)
np.save("./irisArrays/augmentedIrisDataTest.npy", testData)
np.save("./irisArrays/augmentedIrisLabelsTest.npy", testLabels)
