# Чебышев через Fast Fourier Transform
from math import sin, cos, pi
from numpy import linspace
from scipy import fft
import matplotlib.pyplot as plt


dots = 1000
N = 64
A1 = 3
A2 = 2
w1 = 1
w2 = 5
an, bn = [], []
A = []
x = linspace(-1,1,int(dots))
t = linspace(0,2*pi-2*pi/dots,dots)
t2 = linspace(0,2*pi-2*pi/N,N)

def function(x):
  return A1*cos(w1*x)+A2*sin(w2*x)
s = [A1*cos(w1*cos(i))+A2*sin(w2*cos(i)) for i in t]
s2 = [A1*cos(w1*cos(i))+A2*sin(w2*cos(i)) for i in t2]
a = fft.fft(s2) / N * 2

a[0] = a[0]/2
print('Коэффициенты Чебышева:\n', a[:int(N/4)])
sum = []
for i in t:
  temp = 0
  for j in range(int(N/4)):
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