import numpy as np
from sympy import Matrix, pprint

# Задаем нашу матрицу А
A = np.array([[10, 3, 0],
              [3, 15, 1],
              [0, 1, 7]])
# Задаем нашу матрицу f
f = np.array([[2],
              [12],
              [5]])

# Точное решение х уравнения Ax = f
xdef = np.linalg.solve(A,f)
epsilon = 1e-6    # Погрешность, которая нас удовлетворяет
w = 1
Xstart = [[1],[1],[1]]
XJacobi = [Xstart]
XSeidel = [Xstart]
XSOR = [Xstart]
print('A\n')
pprint(Matrix(A))
print('\nf\n')
pprint(Matrix(f))
def criteria(x):
  global xdef
  dif = xdef - x
  # Возвращаем норму
  return np.linalg.norm(dif)/np.linalg.norm(xdef)


# Извлекаем диагональ матрицы А
D = np.diag(np.diag(A))
# Вычисляем обратную матрицу
Dinv = np.linalg.inv(D)
# Создаем матрицы размером А с нулями
Aup = np.zeros(np.shape(A))
Adown = np.zeros(np.shape(A))
for i in range(len(A)):
  for j in range(len(A)):
    if i < j: Aup[i,j] = A[i,j]
    elif i > j: Adown[i,j] = A[i,j]         # А = Аdown + D + Aup

# Якоби
iter = 0
while criteria(XJacobi[iter]) > epsilon:
  x = - Dinv @ Adown @ XJacobi[iter] - Dinv @ Aup @ XJacobi[iter] + Dinv @ f
  XJacobi.append(x)
  iter += 1
print('\nJacobi; Iterations:',iter,'; Error:', criteria(XJacobi[iter]), end='\n\n')
pprint(Matrix(XJacobi[iter]))

# Зейдель
iter = 0
while criteria(XSeidel[iter]) > epsilon:
  x = np.linalg.inv(D + Adown) @ (f - Aup @ XSeidel[iter])
  XSeidel.append(x)
  iter += 1
print('\nSeidel; Iterations:',iter,'; Error:', criteria(XSeidel[iter]), end='\n\n')
pprint(Matrix(XSeidel[iter]))

# SOR
iter = 0
while criteria(XSOR[iter]) > epsilon:
  x = np.linalg.inv(D + w*Adown) @ (w*f - (w*Aup + (w-1)*D) @ XSOR[iter])
  XSOR.append(x)
  iter += 1
print('\nSOR; Iterations:',iter,'; Error:', criteria(XSOR[iter]), end='\n\n')
pprint(Matrix(XSOR[iter]))