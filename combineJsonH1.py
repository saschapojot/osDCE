import numpy as np
import matplotlib.pyplot as plt
import re
import glob
import json


inJsonFileNames=[]
flushNumAll=[]

dataPath="./groupNew5/row0/H1Verify/"
for file in glob.glob(dataPath+"/*.json"):

    matchFlush=re.search(r"flush(\d+)N1",file)
    if matchFlush:
        flushNumAll.append(int(matchFlush.group(1)))
        inJsonFileNames.append(file)
sortedInds=np.argsort(flushNumAll)
sortedFileNames=[inJsonFileNames[ind] for ind in sortedInds]


diffValsAll=np.array([])
for file in sortedFileNames:
    with open(file,"r") as fptr:
        dataTmp=json.load(fptr)
    diffOneFile=np.array(dataTmp["diff"])
    diffValsAll=np.r_[diffValsAll,diffOneFile]

np.savetxt("jsonCombined.txt",diffValsAll,delimiter=",")