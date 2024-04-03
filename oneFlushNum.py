import pickle
import numpy as np
from datetime import datetime
from scipy import sparse
import glob
import re
from multiprocessing import Pool
import pandas as pd
import json
import sys

#This script loads computation results from one flush, and computes observables

#python readCSV.py groupNum rowNum, then parse csv
if len(sys.argv)!=4:
    print("wrong number of arguments")

groupNum=int(sys.argv[1])
rowNum=int(sys.argv[2])
flshNum=int(sys.argv[3])

inParamFileName="./inParamsNew"+str(groupNum)+".csv"
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

inDir="./groupNew"+str(groupNum)+"/row"+str(rowNum)+"/"

pklFile=""

for file in glob.glob(inDir+"/*.pkl"):
    matchFlsh=re.search(r"flush"+str(flshNum)+"N1",file)
    if matchFlsh:
        pklFile=file

matchN1=re.search(r"N1(\d+)N2",pklFile)
N1=int(matchN1.group(1))

matchN2=re.search(r"N2(\d+)L1",pklFile)
N2=int(matchN2.group(1))

matchL1=re.search(r"L1(\d+(\.\d+)?)L2",pklFile)
L1=float(matchL1.group(1))

matchL2=re.search(r"L2(\d+(\.\d+)?)solution",pklFile)
L2=float(matchL2.group(1))

dx1=2*L1/N1
dx2=2*L2/N2
x1ValsAll=np.array([-L1+dx1*n1 for n1 in range(0,N1)])
x2ValsAll=np.array([-L2+dx2*n2 for n2 in range(0,N2)])

tMatStart=datetime.now()

#construct matrices

#construct number operators
#construct H6
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
def avgNc(psiAllInOneFlushj):
    """

    :param psiAllInOneFlush: wavefunctions in one flush
    :param j: jth wavefunction in psiAllInOneFlush
    :return: number of photons for jth wavefunction
    """
    psiAllInOneFlush,j=psiAllInOneFlushj
    Psi=np.reshape(psiAllInOneFlush[j,:,:],-1)
    val=1/2*omegac*np.vdot(Psi,NcMat1@Psi)-1/2*np.vdot(Psi,Psi)+1/omegac*np.vdot(Psi,H6@Psi)
    return np.abs(val)


def avgNm(psiAllInOneFlushj):
    """
    :param psiAllInOneFlush: wavefunctions in one flush
    :param j: jth wavefunction in psiAllInOneFlush
    :return: number of phonons for jth wavefunction
    """
    psiAllInOneFlush,j=psiAllInOneFlushj
    Psi=np.reshape(psiAllInOneFlush[j,:,:],-1)
    val=1/2*omegam*np.vdot(Psi,NmPart1@Psi)-1/2*np.vdot(Psi,Psi)-1/(2*omegam*dx2**2)*np.vdot(Psi,NmPart2@Psi)

    return np.abs(val)


photonNums=[]
phononNums=[]
tLoadStart = datetime.now()
with open(pklFile,"rb") as fptr:
    psiAllInOneFlush = pickle.load(fptr)
tLoadEnd = datetime.now()

print("loading time: ", tLoadEnd - tLoadStart)
p,_,_=psiAllInOneFlush.shape
inVals=[]
for j in range(0,p):
    inVals.append([psiAllInOneFlush, j])

# compute Nc
tNcStart = datetime.now()
for item in inVals:
    photonNums.append(avgNc(item))
tNcEnd = datetime.now()
print("Nc time: ", tNcEnd - tNcStart)


# compute Nm
tNmStart = datetime.now()
for item in inVals:
    phononNums.append(avgNm(item))

tNmEnd = datetime.now()
print("Nm time: ", tNmEnd - tNmStart)
outData={
    "photonNums":photonNums,
    "phononNums":phononNums
}

outDataFileName=inDir+"/numsFlush"+str(flshNum)+".json"
with open(outDataFileName, "w") as fptr:
    json.dump(outData,fptr)