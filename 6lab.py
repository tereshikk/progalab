import numpy as np
from numpy.linalg import norm
from numpy import linspace

A = 5
B = 6
C = 9
D = 3
E = 5
F = 8
R = 10
eps = 1e-4
step = 0.01
dots = 100
mode = 2 # Gradient descent method. 1 stands for rough, 2 for optimized
startPos = np.array([1, 1], dtype='float64')
xy = linspace(-5,5,dots)

function = lambda x, y: 0.5*A*x**2 + B*x*y + 0.5*C*y**2 - D*x - E*y + F
xDerivative1 = lambda x, y: A*x + B*y - D
yDerivative1 = lambda x, y: B*x + C*y - E

# Analytics
xTrue = (B*E-C*D)/(B*B-C*A)
yTrue = (D-A*xTrue)/B
print('Analytical  x:', xTrue,', y: ', yTrue,', z: ', function(xTrue,yTrue))

# Gradient descent
def gradDesc(pos):
  return np.array([xDerivative1(pos[0], pos[1]), yDerivative1(pos[0], pos[1])])

if mode == 2:
  def getNewPos(pos, dir, lval):
    global step
    lpos = pos
    tempPos = np.array([pos[0] - step*dir[0], pos[1] - step*dir[1]])
    tempVal = function(tempPos[0], tempPos[1])
    while tempVal < lval:
      lpos = tempPos
      lval = tempVal
      tempPos = np.array([tempPos[0] - step*dir[0], tempPos[1] - step*dir[1]])
      tempVal = function(tempPos[0], tempPos[1])
    if lval < tempVal:
      return lpos, lval
    else:
      return tempPos, tempVal

  lastPos = startPos
  lastValue = function(lastPos[0], lastPos[1])
  actPos, actVal = getNewPos(lastPos, gradDesc(lastPos), lastValue)
  iterations = 0
  while norm(lastPos - actPos) > eps:
    lastPos = actPos
    lastValue = actVal
    actPos, actVal = getNewPos(lastPos, gradDesc(lastPos), lastValue)
    iterations += 1
else:
  lastPos = startPos
  lastValue = function(lastPos[0], lastPos[1])
  tempPos = np.array([lastPos[0] - step*gradDesc(lastPos)[0], lastPos[1] - step*gradDesc(lastPos)[1]])
  iterations = 0
  while norm(lastPos - tempPos) > eps:
    lastPos = tempPos
    lastValue = function(lastPos[0], lastPos[1])
    tempPos = np.array([lastPos[0] - step*gradDesc(lastPos)[0], lastPos[1] - step*gradDesc(lastPos)[1]])
    iterations += 1
print('Gradient    x:', lastPos[0],', y: ', lastPos[1],', z: ', function(lastPos[0], lastPos[1]), ' Iterations', iterations)

# Newton
lastPos = startPos
lastValue = function(lastPos[0], lastPos[1])
tempPos = np.array([lastPos[0] - xDerivative1(lastPos[0], lastPos[1])/A, lastPos[1] - yDerivative1(lastPos[0], lastPos[1])/C])
iterations = 0
while norm(lastPos - tempPos) > eps:
  lastPos = tempPos
  lastValue = function(lastPos[0], lastPos[1])
  tempPos = np.array([lastPos[0] - xDerivative1(lastPos[0], lastPos[1])/A, lastPos[1] - yDerivative1(lastPos[0], lastPos[1])/C])
  iterations += 1
print('Newton      x:', lastPos[0],', y: ', lastPos[1],', z: ', function(lastPos[0], lastPos[1]), ' Iterations', iterations)

# Graphs
x,y = np.meshgrid(xy,xy)
z = 0.5*A*x**2 + B*x*y + 0.5*C*y**2 - D*x - E*y + F

import plotly.graph_objects as go

fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
fig.update_layout(autosize=False,
                  width=600, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.update_traces(opacity=0.8, selector=dict(type='surface'))
fig.show()