import numpy as np
import pandas as pd
from datetime import datetime
import sys
from scipy.special import hermite
from multiprocessing import Pool

#This script uses operator splitting to compute time evolution


N1=50
N2=30
L1=2
L2=80
dx1=2*L1/N1
dx2=2*L2/N2
x1ValsAll=np.array([-L1+dx1*n1 for n1 in range(0,N1)])
x2ValsAll=np.array([-L2+dx2*n2 for n2 in range(0,N2)])
x1ValsAllSquared=x1ValsAll**2
#python readCSV.py groupNum rowNum, then parse csv
if len(sys.argv)!=3:
    print("wrong number of arguments")

group=int(sys.argv[1])
rowNum=int(sys.argv[2])
inParamFileName="./inParamsNew"+str(group)+".csv"

dfstr=pd.read_csv(inParamFileName)
oneRow=dfstr.iloc[rowNum,:]

j1H=int(oneRow.loc["j1H"])
j2H=int(oneRow.loc["j2H"])

g0=float(oneRow.loc["g0"])
omegam=float(oneRow.loc["omegam"])
omegap=float(oneRow.loc["omegap"])
omegac=float(oneRow.loc["omegac"])
er=float(oneRow.loc["er"])#magnification

thetaCoef=float(oneRow.loc["thetaCoef"])
theta=thetaCoef*np.pi
Deltam=omegam-omegap
e2r=er**2
lmd=(e2r-1/e2r)/(e2r+1/e2r)

# print("j1H"+str(j1H)+"j2H"+str(j2H)+"g0"+str(g0)\
#       +"omegam"+str(omegam)+"omegap"+str(omegap)+"omegac"+str(omegac)+"er"+str(er)+"thetaCoef"+str(thetaCoef))


def H(n,x):
    """

    :param n: order of Hermite polynomial
    :param x:
    :return: value of polynomial at x
    """
    return hermite(n)(x)

#initialize wavefunction

def f1(x1):
    return np.exp(-1/2*omegac*x1**2)*H(j1H,np.sqrt(omegac)*x1)


def f2(x2):
    return np.exp(-1/2*omegam*x2**2)*H(j2H,np.sqrt(omegam)*x2)


f1Vec=np.array([f1(x1) for x1 in x1ValsAll])
f2Vec=np.array([f2(x2) for x2 in x2ValsAll])


psi0=np.outer(f1Vec,f2Vec)
psi0/=np.linalg.norm(psi0,ord=2)
dtEst = 0.002
tFlushStart=0
tFlushStop=0.01
flushNum=5
tTotPerFlush=tFlushStop-tFlushStart

stepsPerFlush=int(np.ceil(tTotPerFlush/dtEst))
dt=tTotPerFlush/stepsPerFlush
timeIndsAll=[]
for fls in range(0,flushNum):
    startingInd = fls * stepsPerFlush
    for j in range(0,stepsPerFlush):
        timeIndsAll.append(startingInd+j)

timeIndsAll=np.array(timeIndsAll)


def evolution1Step(j,psi):
    """

    :param j: time step
    :param psi: wavefunction at the beginning of the time step j
    :return:
    """
    tj=timeIndsAll[j]
    #operator U15
    for n2 in range(0,N2):
        x2n2=x2ValsAll[n2]
        psi[:,n2]*=np.exp(1j*dt*1/2*g0*np.sqrt(2*omegam)*np.cos(omegap*tj)*x2n2)

    #operator U14
    #construct U14
    #construct the exponential part





