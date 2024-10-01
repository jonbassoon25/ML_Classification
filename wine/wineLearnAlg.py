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

xTrain = np.load("./wineArrays/trainData.npy")
yTrain = np.load("./wineArrays/trainLabels.npy")
xTest = np.load("./wineArrays/testData.npy")
yTest = np.load("./wineArrays/testLabels.npy")

print("nearest centroid")
runML(xTrain, yTrain, xTest, yTest, NearestCentroid())

print("k-nearest(n = 3)")
runML(xTrain, yTrain, xTest, yTest, KNeighborsClassifier(n_neighbors=3))

print("naive bayes (geussion)")
runML(xTrain, yTrain, xTest, yTest, GaussianNB())

print("naive bays (multinomia)")
runML(xTrain, yTrain, xTest, yTest, MultinomialNB())

print("decision Tree")
runML(xTrain, yTrain, xTest, yTest, DecisionTreeClassifier())

print("random forest classifier")
runML(xTrain, yTrain, xTest, yTest, RandomForestClassifier(n_estimators=5))

print("SVM (linear)")
runML(xTrain, yTrain, xTest, yTest, SVC(kernel="linear", C=1.0))