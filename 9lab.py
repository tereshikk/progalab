import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# u'' + a1*u' + a2*u = F
# u(0) = c0; u'(0) = c1
a1 = 0
a2 = 1
c0 = 0
c1 = 1
F = lambda x: 1
iterations = 10
h = 0.01
K = lambda x, y: a1 + a2 * (x - y)
f = lambda x: F(x) - c1 * a1 - (c1 * x + c0) * a2

xSpace = np.arange(0, 10, h)
ySpace = f(xSpace)


def wtf(ytemp):
    yt = ytemp
    for i in range(len(xSpace)):
        temp = 0
        for j in range(i + 1):
            temp += 2 * K(xSpace[j], xSpace[i]) * ytemp[j]
        temp += -K(xSpace[i], xSpace[0]) * ytemp[0] - K(xSpace[i], xSpace[i]) * ytemp[i]
        yt[i] = f(xSpace[i]) + temp * h / 2
    return yt


yt = wtf(ySpace)
for i in range(iterations):
    ySpace = np.copy(yt)
    yt = wtf(ySpace)
ytemp = np.copy(yt)
for i in range(len(xSpace)):
    temp = 0
    for j in range(i):
        temp += 2 * (xSpace[i] - xSpace[j]) * ytemp[j]
    temp += -(xSpace[0] - xSpace[i]) * ytemp[0]
    yt[i] = c0 + c1 * xSpace[i] + temp * h / 2


# plt.plot(xSpace,yt)
# plt.show()

def equation(u, x):
    return (u[1], -u[1] * a1 - u[0] * a2 + F(x))


solution = odeint(equation, [0, 1], xSpace)[:, 0]
# plt.plot(xSpace, solution)
# plt.show()
fig, ax = plt.subplots()
fig.set_size_inches((20, 9))
ax.plot(xSpace, yt, '-', label='Numerical solution')
ax.plot(xSpace, solution, '--', label='Exact solution')
leg = ax.legend()
fig.show()
plt.show()
# plt.plot(xSpace, ytemp)
# plt.show()
difff = []
for i in range(len(xSpace)):
    difff.append(abs(yt[i] - solution[i]))
plt.plot(xSpace, difff)
plt.show()