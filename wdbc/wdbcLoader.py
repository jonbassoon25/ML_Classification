import numpy as np
import matplotlib.pyplot as plt

with open("./data/wdbc.data") as wdbcData:
	wdbcLines = wdbcData.readlines()
   
wdbcLines = wdbcLines[:-1]
diagnosesTypes = ["B", "M"]
diagnoses = [diagnosesTypes.index(line.split(",")[1]) for line in wdbcLines]

data = [line.split(",")[2:] for line in wdbcLines]

diagnoses = np.array(diagnoses, dtype="uint8")
data = np.array(data, dtype="double")

np.save("./wdbcArrays/wdbcData.npy", data)
np.save("./wdbcArrays/wdbcDiagnoses.npy", diagnoses)
