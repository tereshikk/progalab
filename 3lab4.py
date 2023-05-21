# Чебышев через интегралы
from math import sin, cos, pi
from numpy import linspace
from scipy import integrate, fft
import matplotlib.pyplot as plt


dots = 512
N = 32
A1 = 1
A2 = 2
w1 = 3
w2 = 4
an, bn = [], []
A = []
x = linspace(-1,1,int(dots))
t = linspace(pi,2*pi,dots)
t2 = linspace(pi,2*pi,N)

def function(x):
  return A1*cos(w1*cos(x))+A2*sin(w2*cos(x))

def func2(x,n):
  return (A1*cos(w1*cos(x))+A2*sin(w2*cos(x)))*cos(n*x)

s = [A1*cos(w1*cos(i))+A2*sin(w2*cos(i)) for i in t]
s2 = [A1*cos(w1*cos(i))+A2*sin(w2*cos(i)) for i in t2]
a = fft.fft(s2) / N
a = a.real
a = [integrate.quad(function,0,pi)[0]/pi]
for n in range(1,N+1):
  a.append(integrate.quad(lambda x: (A1*cos(w1*cos(x))+A2*sin(w2*cos(x)))*cos(n*x),0,pi)[0]/pi*2)



print(len(a),a)
sum = []
for i in t:
  temp = 0
  for j in range(N+1):
    temp += a[j]*cos(j*i)
  sum.append(temp)
difference = [abs(sum[i] - s[i]) for i in range(dots)]
#t = t[int((len(t)/2)):]
#s = s[int((len(s)/2)):]
#sum = sum[int((len(sum)/2)):]
#difference = difference[int((len(difference)/2)):]

plt.plot(x,difference)
plt.show()
plt.plot(x,sum,x,s)
plt.show()