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
import matplotlib.pyplot as plt

#This script verifies exp(-dt H3) part of the algorithm

if len(sys.argv)!=3:
    print("wrong number of arguments")

group=int(sys.argv[1])
rowNum=int(sys.argv[2])
inParamFileName="./inParamsNew"+str(group)+".csv"

outDir="./groupNew"+str(group)+"/row"+str(rowNum)+"/"
Path(outDir).mkdir(parents=True, exist_ok=True)
#read parameters from csv
dfstr=pd.read_csv(inParamFileName)
oneRow=dfstr.iloc[rowNum,:]

j1H=int(oneRow.loc["j1H"])
j2H=int(oneRow.loc["j2H"])

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
height=np.exp(-1/4)*1.1
upperBoundL1=(1/(2*omegac))**(1/2)
L1=1.1*(-2*np.log(height)/omegac)**(1/2)


L2=5
N2=500

N1=100

dx1=2*L1/N1

dx2=2*L2/N2
print("dx1="+str(dx1))

x1ValsAll=np.array([-L1+dx1*n1 for n1 in range(0,N1)])
x2ValsAll=np.array([-L2+dx2*n2 for n2 in range(0,N2)])
x1ValsAllSquared=x1ValsAll**2
x2ValsAllSquared=x2ValsAll**2

k1ValsAll=[]
for n1 in range(0,int(N1/2)):
    k1ValsAll.append(2*np.pi/(2*L1)*n1)
for n1 in range(int(N1/2),N1):
    k1ValsAll.append(2*np.pi/(2*L1)*(n1-N1))
k1ValsAll=np.array(k1ValsAll)
k1ValsSquared=k1ValsAll**2
k2ValsAll=[]
for n2 in range(0,int(N2/2)):
    k2ValsAll.append(2*np.pi/(2*L2)*n2)
for n2 in range(int(N2/2),N2):
    k2ValsAll.append(2*np.pi/(2*L2)*(n2-N2))
k2ValsAll=np.array(k2ValsAll)
k2ValsSquared=k2ValsAll**2
dtEst = 0.000025
tFlushStart=0
tFlushStop=0.001
flushNum=300
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


# f1Vec=np.array([f1(x1) for x1 in x1ValsAll])
# f2Vec=np.array([f2(x2) for x2 in x2ValsAll])
#
#
# psi0=np.outer(f1Vec,f2Vec)
# psi0/=np.linalg.norm(psi0,ord=2)
# psi0=np.array(psi0,dtype=complex)
def psiExactElem(n1, n2, t):
    x1Tmp2 = x1ValsAllSquared[n1]
    x2Tmp = x2ValsAll[n2]
    part1 = np.exp(-1j / omegap * (np.cos(omegap * t) - 1))

    part2 = np.exp(1j * x2Tmp / (-g0 * omegac * np.sqrt(2 / omegam) * x1Tmp2 + 1 / 2 * g0 * np.sqrt(2 / omegam)))

    return part1 * part2


def psiExact(t):
    psiAnalytical = np.zeros((N1, N2), dtype=complex)
    for n1 in range(0, N1):
        for n2 in range(0, N2):
            psiAnalytical[n1, n2] = psiExactElem(n1, n2, t)
    psiAnalytical /= np.linalg.norm(psiAnalytical, ord=2)
    return psiAnalytical

psi0=psiExact(0)

#plot init wavefunction
plt.figure()
plt.imshow(np.abs(psi0))
plt.title("$\psi_{0}$")
plt.colorbar()
plt.savefig(outDir+"psi0.png")
plt.close()

def f(n1,t):
    """

    :param n1: index for x1n1
    :param t: time
    :return: coefficient for evolution using H3
    """
    x1n1Squared=x1ValsAllSquared[n1]

    val= -g0*omegac*np.sqrt(2/omegam)*np.sin(omegap*t)*x1n1Squared\
            +1/2*g0*np.sqrt(2/omegam)*np.sin(omegap*t)
    return val
def evolution1Step(j,psi):
    """

    :param j: time step
    :param psi: wavefunction at the beginning of the time step j
    :return:
    """
    tj=timeValsAll[j]
    t1StepStart=datetime.now()
    ###################### exp(-idt H3)
    W = np.fft.fft(psi, axis=1, norm="ortho")

    fx1n1Vec = [f(n1, tj) for n1 in range(0, N1)]
    matTmp = np.array(np.outer(fx1n1Vec, k2ValsAll), dtype=complex)
    matTmp *= 1j * dt

    M = np.exp(matTmp)

    W = W * M

    psi = np.fft.ifft(W, axis=1, norm="ortho")
    t1StepEnd = datetime.now()
    # print("1 step time: ", t1StepEnd - t1StepStart)

    return psi


#construct matrices

#construct number operators
#construct H6
tMatStart=datetime.now()

leftMat=sparse.diags(-2*np.ones(N1),offsets=0,format="csc",dtype=complex)\
    +sparse.diags(np.ones(N1-1),offsets=1,format="csc",dtype=complex)\
    +sparse.diags(np.ones(N1-1),offsets=-1,format="csc",dtype=complex)

H6=-1/(2*dx1**2)*sparse.kron(leftMat,sparse.eye(N2,dtype=complex,format="csc"),format="csc")
#compute <Nc>
tmp0=sparse.diags(x1ValsAll**2,format="csc")
IN2=sparse.eye(N2,dtype=complex,format="csc")
NcMat1=sparse.kron(tmp0,IN2)
# compute Nm
S2=sparse.diags(np.power(x2ValsAll,2),format="csc")
Q2=sparse.diags(-2*np.ones(N2),offsets=0,format="csc",dtype=complex)\
    +sparse.diags(np.ones(N2-1),offsets=1,format="csc",dtype=complex)\
    +sparse.diags(np.ones(N2-1),offsets=-1,format="csc",dtype=complex)
IN1=sparse.eye(N1,dtype=complex,format="csc")
NmPart1=sparse.kron(IN1,S2)
NmPart2=sparse.kron(IN1,Q2)

tMatEnd=datetime.now()
print("construct mat time: ",tMatEnd-tMatStart)


def avgNc(Psi):
    """

    :param Psi: wavefunction
    :return: number of photons for wavefunction
    """
    psiVec=np.reshape(Psi,-1)
    val=1/2*omegac*np.vdot(psiVec,NcMat1@psiVec)-1/2*np.vdot(psiVec,psiVec)+1/omegac*np.vdot(psiVec,H6@psiVec)
    return np.abs(val)

def avgNm(Psi):
    """
    :param Psi: wavefunction
    :return: number of phonons for wavefunction
    """
    psiVec=np.reshape(Psi,-1)
    val=1/2*omegam*np.vdot(psiVec,NmPart1@psiVec)-1/2*np.vdot(psiVec,psiVec)-1/(2*omegam*dx2**2)*np.vdot(psiVec,NmPart2@psiVec)

    return np.abs(val)



def oneFlush(psiIn,fls):
    """

    :param psiIn: starting value of the wavefunction in one flush
    :param fls:
    :return: starting value of the wavefunction in the next flush
    """
    startingInd = fls * stepsPerFlush

    # psiMat=np.zeros((stepsPerFlush+1,N1,N2),dtype=complex)

    # psiMat[0,:,:]=copy.deepcopy(psiIn)
    psiCurr=copy.deepcopy(psiIn)
    photonPerFlush=[avgNc(psiCurr)]
    phononPerFlush=[avgNm(psiCurr)]
    diffPerFlush=[np.linalg.norm(psiCurr-psiExact(dt*startingInd),ord=2)]
    for j in range(0,stepsPerFlush):
        indCurr=startingInd+j

        psiNext=evolution1Step(indCurr,psiCurr)
        psiNextExact=psiExact(dt*(indCurr+1))
        diffPerFlush.append(np.linalg.norm(psiNext-psiNextExact,ord=2))
        psiCurr=copy.deepcopy(psiNext)
        photonPerFlush.append(avgNc(psiCurr))
        phononPerFlush.append(avgNm(psiCurr))

    outFile = outDir + "flush" + str(fls) + "N1" + str(N1)\
              +"N2" + str(N2) + "L1" + str(L1)\
              +"L2" + str(L2) + "num.json"
    outData={
        "photonNums":photonPerFlush,
        "phononNums":phononPerFlush,
        "diff": diffPerFlush
    }
    with open(outFile, "w") as fptr:
        json.dump(outData, fptr)

    return psiCurr


outPsiInit = outDir + "init"  + "N1" + str(N1)\
              +"N2" + str(N2) + "L1" + str(L1)\
              +"L2" + str(L2) + "solution.pkl"
outPsiFinal = outDir + "final"  + "N1" + str(N1)\
              +"N2" + str(N2) + "L1" + str(L1)\
              +"L2" + str(L2) + "solution.pkl"
#evolution
psiStart=copy.deepcopy(psi0)
with open(outPsiInit,"wb") as fptr:
    pickle.dump(psiStart,fptr,protocol=pickle.HIGHEST_PROTOCOL)
for fls in range(0,flushNum):
    tFlsStart=datetime.now()
    psiFinal=oneFlush(psiStart,fls)
    tFlsEnd=datetime.now()
    print("one flush time: ",tFlsEnd-tFlsStart)
    psiStart=copy.deepcopy(psiFinal)


with open(outPsiFinal,"wb") as fptr:
    pickle.dump(psiStart,fptr,protocol=pickle.HIGHEST_PROTOCOL)