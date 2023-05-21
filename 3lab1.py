# Фурье
from math import sin, cos, pi
from numpy import linspace
from scipy import integrate
import matplotlib.pyplot as plt


dots = 100
N = 64
A1 = 3
A2 = 2
w1 = 1
w2 = 5
an, bn = [], []

def function(x):
  return A1*cos(w1*x)+A2*sin(w2*x)

s = [A1*cos(w1*x)+A2*sin(w2*x) for x in linspace(-1,1,dots)]
a0 = integrate.quad(func=function,a=-1,b=1)[0]
for n in range(N):
  an.append(integrate.quad(func=(lambda x: function(x)*cos(pi*(n+1)*x)),a=-1,b=1)[0])
  bn.append(integrate.quad(func=(lambda x: function(x)*sin(pi*(n+1)*x)),a=-1,b=1)[0])

sum = []
for x in linspace(-1,1,dots):
  temp = 0
  for n in range(N):
    temp += an[n]*cos(pi*(n+1)*x) + bn[n]*sin(pi*(n+1)*x)
  sum.append(a0/2 + temp)
#sum

difference = [abs(sum[i] - s[i]) for i in range(dots)]
print(sum, s, sep='\n\n')

plt.plot(linspace(-1,1,dots),difference)
plt.show()
plt.plot(linspace(-1,1,dots),s)
plt.plot(linspace(-1,1,dots),sum)
plt.show()
