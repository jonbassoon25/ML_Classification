import csv
import numpy as np

wineData = []
wineLabels = []
with open("./wineData/winequality-white.csv", newline='') as csvFile:
    reader = csv.reader(csvFile, delimiter=";")
    i = 0
    for row in reader:
        if i == 0:
            i += 1
            continue
        wineData.append([float(i) for i in row])
        wineLabels.append(wineData[-1].pop(-1))
        
wineData = np.array(wineData)
wineLabels = np.array(wineLabels, dtype="uint8")

np.save("./wineArrays/wineData.npy", wineData)
np.save("./wineArrays/wineLabels.npy", wineLabels)