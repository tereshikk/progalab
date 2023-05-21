# Чебышев через суммы
from math import sin, cos, pi
from numpy import linspace
from scipy import fft
import matplotlib.pyplot as plt


dots = 128
N = 10
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
  return A1*cos(w1*x)+A2*sin(w2*x)
s = [A1*cos(w1*cos(i))+A2*sin(w2*cos(i)) for i in t]
s2 = [A1*cos(w1*cos(i))+A2*sin(w2*cos(i)) for i in t2]
a = fft.fft(s2) / N
a = a.real
a = []
temp = 0
for i in range(N-1):
  temp += s2[i]
a.append(temp/N)
for n in range(N):
  temp = 0
  for i in range(N-1):
    temp += s2[i]*cos((n+1)*t2[i])
  a.append(2*temp/N)



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