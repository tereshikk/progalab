from sympy.matrices import Matrix
from sympy import diff, pprint
from sympy.utilities.lambdify import lambdify
from sympy.abc import symbols

m = Matrix([[16, 3, 2],
            [3, 5, 1],
            [2, 1, 10]])
# m = Matrix([[5,5,3,4],[5,7,8, 6],[3, 8, 3, 2],[4, 6, 2, 1]])
eigenValues = [10 * i for i in range(0, len(m[0, :]))]
iter1 = 40
iter2 = 40
eps = 1.0e-5
tau = 0.5
lamda = symbols('lamda')
chareq = m.charpoly(lamda)
chareqdev = diff(chareq)

fchareq = lambdify(lamda, chareq.as_expr())
fchareqdev = lambdify(lamda, chareqdev.as_expr())

for _ in range(iter1):
    for i in range(len(eigenValues)):
        eigenValues[i] = eigenValues[i] - fchareq(eigenValues[i]) / fchareqdev(eigenValues[i])
print('Собственные значения матрицы:\n')
pprint(Matrix(eigenValues))

######
import numpy as np
from numpy.linalg import eig


A = np.array(m, dtype='float64')
A11 = np.copy(A)


def criteria(eigenVector, eigenValue):
    global A11
    return np.linalg.norm(((A11 - (np.eye(len(A11)) * eigenValue)).dot(eigenVector)).astype('float64'))


f = np.array([[0] for _ in range(len(A))])
for eigenValue in eigenValues:
    A = np.array(m, dtype='float64')
    for i in range(len(A)):
        A[i, i] = A[i, i] - eigenValue
    w = 1
    Xstart = np.array([[1] for _ in range(len(A))], dtype='float64')
    XJacobi = np.copy(Xstart)
    XSeidel = np.copy(Xstart)
    XSOR = np.copy(Xstart)
    iterations = [0, 0, 0]

    while criteria(XJacobi, eigenValue) > eps:  # Jacobi iter
        xn1 = [1 for _ in range(len(A))]
        for i in range(len(A)):
            axn = 0
            for j in range(len(A)):
                axn += A[i, j] * XJacobi[j]
            xn1[i] = XJacobi[i] + (-axn + f[i]) / A[i, i] * tau
        XJacobi = xn1
        iterations[0] += 1

    while criteria(XSeidel, eigenValue) > eps:  # Seidel iter
        xn1 = [1 for _ in range(len(A))]
        for i in range(len(A)):
            a1xn = 0
            for j in range(i):
                a1xn += A[i, j] * xn1[j]
            a2xn = 0
            for j in range(i + 1, len(A)):
                a2xn += A[i, j] * XSeidel[j]
            xn1[i] = (-a1xn - a2xn + f[i]) / A[i, i]
        XSeidel = xn1
        iterations[1] += 1

    while criteria(XSOR, eigenValue) > eps:  # SOR iter
        xn1 = [1 for _ in range(len(A))]
        for i in range(len(A)):
            a1xn = 0
            for j in range(i):
                a1xn += A[i, j] * xn1[j]
            a2xn = 0
            for j in range(i + 1, len(A)):
                a2xn += A[i, j] * XSOR[j]
            xn1[i] = w * (-a1xn - a2xn + f[i]) / A[i, i] + (1 - w) * XSOR[i]
        XSOR = xn1
        iterations[2] += 1
    coef1 = np.linalg.norm(XJacobi)
    coef2 = np.linalg.norm(XSeidel)
    coef3 = np.linalg.norm(XSOR)
    for i in range(len(XJacobi)):
        XJacobi[i] = abs(float(XJacobi[i] / coef1))
        XSeidel[i] = abs(float(XSeidel[i] / coef2))
        XSOR[i] = abs(float(XSOR[i] / coef3))
    print('\n', 'Eigen value:', eigenValue)
    print('Jacobi; Iters: ', iterations[0], '\n')
    pprint(Matrix(XJacobi))
    print('Seidel; Iters: ', iterations[1], '\n')
    pprint(Matrix(XSeidel))
    print('SOR; Iters: ', iterations[2], '\n')
    pprint(Matrix(XSOR))
    print('---------------')
truevalues = eig(np.array(m, dtype='float64'))
print('\nTrue eigen values\n')
pprint(Matrix(truevalues[0]))
print('\n\nTrue vectors\n')
pprint(Matrix(truevalues[1]))