import numpy as np
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def runML(xTrain, yTrain, xTest, yTest, clf):
    clf.fit(xTrain, yTrain)
    print("Predictions:\t", clf.predict(xTest))
    print("Actual Labels:\t", yTest)
    print(f"Score: {clf.score(xTest, yTest)}")
    print()

x = np.load("./irisArrays/irisData.npy")
y = np.load("./irisArrays/irisLabels.npy")

idx = np.argsort(np.random.random(y.shape[0]))
x = x[idx]
y = y[idx]

split = 120
xTrain = x[:split]
yTrain = y[:split]
xTest = x[split:]
yTest = y[split:]

xa = np.load("./irisArrays/augmentedIrisData.npy")
ya = np.load("./irisArrays/augmentedIrisLabels.npy")
xaTest = np.load("./irisArrays/augmentedIrisDataTest.npy")
yaTest = np.load("./irisArrays/augmentedIrisLabelsTest.npy")

print("nearest centroid")
runML(xTrain, yTrain, xTest, yTest, NearestCentroid())

print("k-nearest(n = 3)")
runML(xTrain, yTrain, xTest, yTest, KNeighborsClassifier(n_neighbors=3))

print("naive bayes (geussion)")
runML(xTrain, yTrain, xTest, yTest, GaussianNB())

print("decision Tree")
runML(xTrain, yTrain, xTest, yTest, DecisionTreeClassifier())

print("random forest classifier")
runML(xa, ya, xaTest, yaTest, RandomForestClassifier(n_estimators=5))

print("SVM (linear)")
runML(xa, ya, xaTest, yaTest, SVC(kernel="linear", C=1.0))
