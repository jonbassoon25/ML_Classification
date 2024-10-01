import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition

def generateData(pca, trainData, n):
    original = pca.components_.copy()
    ncomp = pca.components_.shape[0]
    a = pca.transform(trainData)
    for i in range(n, ncomp):
        pca.components_[i,:] += np.random.normal(scale = 0.1, size = ncomp)
    b = pca.inverse_transform(a) #returns the data in it's original format instead of being sorted
    pca.components_ = original.coppy()
    return b

#load wdbc arrays
diagnosis = np.load("./wdbcArrays/wdbcDiagnosis.npy")
data = np.load("./wdbcArrays/wdbcData.npy")

pca = decomposition.PCA(30)
pca.fit(data)
#print(pca.explained_variance_ratio_)
#1 important value


#shuffle data
idx = np.argsort(np.random.random(diagnosis.shape[0])) #sort random values into indicies
shuffledData = data[idx] #reassign indexes of data
shuffledDiagnosis = diagnosis[idx] #reassign indexes of labels

split = 60
testDiagnosis = shuffledDiagnosis[:split]
trainDiagnosis = shuffledDiagnosis[split:]
testData = shuffledData[:split]
trainData = shuffledData[split:]

np.save("./wdbcArrays/shuffledWdbcData", shuffledData)
np.save("./wdbcArrays/shuffledWdbcDiagnosis", shuffledDiagnosis)

pca.fit(shuffledData)

#make augmented data
augDiagnosis = np.zeros(len(trainDiagnosis) * 10, dtype="uint8")
augData = np.zeros((len(trainDiagnosis) * 10, 30))

for i in range(10):
    if i == 0:
        augDiagnosis[0:trainData.shape[0]] = trainDiagnosis
        augData[0:trainData.shape[0],:] = trainData
    else:
        augDiagnosis[i * trainData.shape[0] : i * trainData.shape[0] + trainData.shape[0]] = trainDiagnosis
        augData[i * trainData.shape[0] : i * trainData.shape[0] + trainData.shape[0],:] = generateData(pca, trainData, 1)

idx = np.argsort(np.random.random(augData.shape[0]))
q = augData[idx]
r = augDiagnosis[idx]
np.save("./wdbcArrays/augmentedWdbcData.npy", q)
np.save("./wdbcArrays/augmentedWdbcDiagnosis.npy", r)
np.save("./wdbcArrays/augmentedWdbcDataTest.npy", testData)
np.save("./wdbcArrays/augmentedWdbcDiagnosisTest.npy", testDiagnosis)
