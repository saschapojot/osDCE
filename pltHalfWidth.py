import numpy as np
import matplotlib.pyplot as plt



height=1/2
nExp=[1,2,3,4]
omegacVals=[10**n for n in nExp]
L1=5
def width(omegac):
    return np.sqrt(-2*np.log(height)/omegac)

widthValsAll=np.array([width(omegacVals[0]),width(omegacVals[1]),width(omegacVals[2])])
widthValsAll*=2
print(widthValsAll)

minGridValsAll=[val/20 for val in widthValsAll]
N1ValsAll=[2*L1/val for val in minGridValsAll]
print(N1ValsAll)



xValsAll=np.linspace(-5,5,10000)

funcVals0=[np.exp(-1/2*omegacVals[0]*x**2) for x in xValsAll]
funcVals1=[np.exp(-1/2*omegacVals[1]*x**2) for x in xValsAll]
funcVals2=[np.exp(-1/2*omegacVals[2]*x**2) for x in xValsAll]

plt.figure()
plt.plot(xValsAll,funcVals0,color="blue",label="$\omega_{c}=10^1$")
plt.plot(xValsAll,funcVals1,color="red",label="$\omega_{c}=10^2$")

plt.plot(xValsAll,funcVals2,color="green",label="$\omega_{c}=10^3$")
plt.xlabel("$x$")
plt.ylabel("$e^{-\\frac{1}{2}\omega_{c}x^2}$")
plt.title("$\exp(-\\frac{1}{2}\omega_{c}x^2)$")

plt.legend(loc="best")
plt.savefig("widthOmegac.png")

