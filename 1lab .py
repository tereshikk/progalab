import numpy as np
from sympy import Matrix, pprint


# Ax = f
L = np.zeros((3,3))
# Задаем нашу матрицу А|f
A = np.array([[2,1,3,10],
              [11,7,5,2],
              [9,8,4,6]], dtype='float64')
# Задаем f
f = A[:,-1].reshape(len(A),1)
print('A|f = \n')
pprint(Matrix(A))
# Приведение к треугольному виду
for i in range(len(A)-1):
  L[i:,i] = A[i:,i]
  A[i] = A[i]/A[i,i]
  for j in range(len(A)-i-1):
    A[i+j+1] = A[i+j+1] - A[i]*A[i+j+1,i]
L[-1,-1] = A[-1,-2]
A[-1] = A[-1]/A[-1,-2]
U = np.copy(A[:,:-1])
print('Ux = \n')        # Ux = y
pprint(Matrix(A))

for i in range(len(A)-1):
  for j in range(len(A)-1-i):
    A[-i-j-2] = A[-i-j-2] - A[-i-1]*A[-i-j-2,-i-2]
print('x = \n')
pprint(Matrix(A))
print('U = \n')
pprint(Matrix(U))
print('L = \n')        # Ly = f
pprint(Matrix(L))
print('LU = \n')       # Ax = f, Ux = y, Ly = f   => Ax = Ly = LUx   => A = LU
pprint(Matrix(L @ U))