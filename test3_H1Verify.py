import numpy as np
import pandas as pd
from datetime import datetime
import sys
from scipy.special import hermite
import copy
import pickle
from pathlib import Path
from scipy import sparse
import json
from scipy.linalg import expm
#this script verifies H1
# python readCSV.py groupNum rowNum, then parse csv
if len(sys.argv)!=3:
    print("wrong number of arguments")

group=int(sys.argv[1])
rowNum=int(sys.argv[2])
inParamFileName="./inParamsNew"+str(group)+".csv"
#read parameters from csv
dfstr=pd.read_csv(inParamFileName)
oneRow=dfstr.iloc[rowNum,:]

# j1H=int(oneRow.loc["j1H"])
# j2H=int(oneRow.loc["j2H"])

g0=float(oneRow.loc["g0"])
omegam=float(oneRow.loc["omegam"])
omegap=float(oneRow.loc["omegap"])
omegac=float(oneRow.loc["omegac"])
er=float(oneRow.loc["er"])#magnification
r=np.log(er)
thetaCoef=float(oneRow.loc["thetaCoef"])
theta=thetaCoef*np.pi
Deltam=omegam-omegap
e2r=er**2
lmd=(e2r-1/e2r)/(e2r+1/e2r)*Deltam


# print("j1H"+str(j1H)+"j2H"+str(j2H)+"g0"+str(g0)\
#       +"omegam"+str(omegam)+"omegap"+str(omegap)+"omegac"+str(omegac)+"er"+str(er)+"thetaCoef"+str(thetaCoef))


N2=10000
height1=1/2
width1=(-2*np.log(height1)/omegac)**(1/2)
minGrid1=width1/50

L1=5
L2=80

N1=int(np.ceil(L1*2/minGrid1))
if N1 %2==1:
    N1+=1
print("N1="+str(N1))
# N1=9000
dx1=2*L1/N1
dx2=2*L2/N2
print("minGrid1="+str(minGrid1))
print("dx1="+str(dx1))
print("dx2="+str(dx2))

x1ValsAll=np.array([-L1+dx1*n1 for n1 in range(0,N1)])
x2ValsAll=np.array([-L2+dx2*n2 for n2 in range(0,N2)])
x1ValsAllSquared=x1ValsAll**2
x2ValsAllSquared=x2ValsAll**2

# k1ValsAll=[]
# for n1 in range(0,int(N1/2)):
#     k1ValsAll.append(2*np.pi/(2*L1)*n1)
# for n1 in range(int(N1/2),N1):
#     k1ValsAll.append(2*np.pi/(2*L1)*(n1-N1))
# k1ValsAll=np.array(k1ValsAll)
# k1ValsSquared=k1ValsAll**2


# k2ValsAll=[]
# for n2 in range(0,int(N2/2)):
#     k2ValsAll.append(2*np.pi/(2*L2)*n2)
# for n2 in range(int(N2/2),N2):
#     k2ValsAll.append(2*np.pi/(2*L2)*(n2-N2))
# k2ValsAll=np.array(k2ValsAll)
# k2ValsSquared=k2ValsAll**2



#construct the space part in the analytical solution for H1

matSpace=np.zeros((N1,N2),dtype=complex)

for n1 in range(0,N1):
    for n2 in range(0,N2):
        x1SqTmp=x1ValsAllSquared[n1]
        x2Tmp=x2ValsAll[n2]
        matSpace[n1,n2]=1/2*g0*np.sqrt(2*omegam)*x2Tmp-g0*omegac*np.sqrt(2*omegam)*x1SqTmp*x2Tmp

matSpace*=1j/omegap

def psiAnalytical(t):
    psiTmp=np.exp(matSpace*np.sin(omegap*t))
    psiTmp/=np.linalg.norm(psiTmp,"fro")
    return psiTmp

dtEst = 1e-4
tFlushStart=0
tFlushStop=0.001
flushNum=4000
tTotPerFlush=tFlushStop-tFlushStart

stepsPerFlush=int(np.ceil(tTotPerFlush/dtEst))
dt=tTotPerFlush/stepsPerFlush

timeValsAll=[]
for fls in range(0,flushNum):
    startingInd = fls * stepsPerFlush
    for j in range(0,stepsPerFlush):
        timeValsAll.append(startingInd+j)
# print(timeValsAll)
timeValsAll=np.array(timeValsAll)*dt
outDir="./groupNew"+str(group)+"/row"+str(rowNum)+"/test3_H1Verify/"
Path(outDir).mkdir(parents=True, exist_ok=True)
def evolution1Step(j,psi):
    """

    :param j: time step
    :param psi: wavefunction at the beginning of the time step j
    :return:
    """
    tj=timeValsAll[j]
    # t1StepStart=datetime.now()
    ######################## exp(-idt H1)
    #operator U15
    for n2 in range(0,N2):
        x2n2=x2ValsAll[n2]
        psi[:,n2]*=np.exp(1j*dt*1/2*g0*np.sqrt(2*omegam)*np.cos(omegap*tj)*x2n2)

    #operator U14
    #construct U14
    #construct the exponential part
    U14=np.array(np.outer(x1ValsAllSquared,x2ValsAll),dtype=complex)
    # print("shape U14: "+str(U14.shape))
    U14*=-1j*dt*g0*omegac*np.sqrt(2*omegam)*np.cos(omegap*tj)
    U14=np.exp(U14)
    psi=np.multiply(U14,psi)
    # print("norm U14 squared="+str(np.linalg.norm(U14,"fro")**2))

    # print("norm psi="+str(np.linalg.norm(psi,"fro")))

    return psi


tEvoStart=datetime.now()

psiNumericalCurr=psiAnalytical(0)

psiAnaCurr=psiAnalytical(0)

for fls in range(0,flushNum):
    tFlushStart = datetime.now()
    startingInd = fls * stepsPerFlush
    diffPerFlush=[]
    for st in range(0,stepsPerFlush):
        j=startingInd+st
        psiNumericalNext=evolution1Step(j,psiNumericalCurr)
        psiNumericalCurr=psiNumericalNext
        psiAnaCurr=psiAnalytical(timeValsAll[j]+dt)
        diffTmp=np.linalg.norm(psiNumericalCurr-psiAnaCurr,ord="fro")
        # print("psiNumericalCurr norm="+str(np.linalg.norm(psiNumericalCurr,"fro")))
        diffPerFlush.append(diffTmp)
    outData={"diff":diffPerFlush}
    outFile = outDir + "flush" + str(fls) + "N1" + str(N1) \
              + "N2" + str(N2) + "L1" + str(L1) \
              + "L2" + str(L2) + "diff.json"

    with open(outFile,"w") as fptr:
        json.dump(outData,fptr)

    tFlushEnd = datetime.now()
    print("flush "+str(fls)+" time: ",tFlushEnd-tFlushStart)


tEvoEnd=datetime.now()
print("evo time: ",tEvoEnd-tEvoStart)