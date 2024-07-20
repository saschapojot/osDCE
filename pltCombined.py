import numpy as np
import matplotlib.pyplot as plt


inFile="./jsonCombined.txt"

arr=np.loadtxt(inFile)
dtEst = 0.0001
tFlushStart=0
tFlushStop=0.001
flushNum=4000
tTotPerFlush=tFlushStop-tFlushStart

stepsPerFlush=int(np.ceil(tTotPerFlush/dtEst))
dt=tTotPerFlush/stepsPerFlush
# print("dt="+str('{:.2e}'.format(dt)))

tValsAll=np.array(range(1,len(arr)+1))*dt

length=len(arr)
seg=int(length*1/4)
# print(np.where(arr>0.8)[0][0])

plt.figure()
plt.plot(tValsAll[:seg],arr[:seg],color="black")
plt.savefig("tmp.png")