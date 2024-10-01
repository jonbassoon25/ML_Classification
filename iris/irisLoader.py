import numpy as np
import matplotlib.pyplot as plt

with open("./data/iris.data") as irisData:
	irisLines = irisData.read()

irisLines = irisLines.strip()
irisLines = irisLines.split("\n")
names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
labels = []
for i in range(len(irisLines)):
	irisLines[i] = irisLines[i].split(",")
	for j in range(4):
		irisLines[i][j] = float(irisLines[i][j])
	labels.append(names.index(irisLines[i][4]))
	irisLines[i] = irisLines[i][:-1]

labels = np.array(labels, dtype="uint8")
data = np.array(irisLines, dtype="double")
zScores = (data - data.mean(axis=0)) / data.std(axis=0)

np.save("./irisArrays/irisLabels.npy", labels)
np.save("./irisArrays/irisData.npy", data)
np.save("./irisArrays/irisZScores.npy", zScores)
