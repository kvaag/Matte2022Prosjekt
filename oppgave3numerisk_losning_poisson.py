# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 12:47:20 2026

@author: leknu
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la


# ------------------------------------------------------------
# Bygging av Laplace-matrisen med løkke:
m = 50
x0,x1 = -1.0,1.0
x = np.linspace(x0, x1, m + 2)
h = (x1-x0)/(m+1)  
# Initialiserer matrisen med nuller
L = np.zeros((m, m))

# Løkke over radene
for i in range(m - 1):
    L[i, i]     = -2
    L[i, i + 1] =  1
    L[i + 1, i] =  1

# Siste diagonalelement (nederst til høyre)
L[-1, -1] = -2

# Skalerer med 1/h^2 (kommer fra sentraldifferansen)
L = L / h**2
print(L)

def f(x):
    return np.cos(np.pi*x)

F = f(x[1:-1])

#Randbetingelser
venstre = 0
hoyre = 2
F[0] -= venstre/h**2
F[-1] -= hoyre/h**2

U = la.solve(L,F)

u_num = np.zeros(m + 2)
u_num[0] = venstre
u_num[-1] = hoyre 
u_num[1:-1] = U


u_eksakt = -(1/np.pi**2)*np.cos(np.pi*x) + x + 1 - 1/np.pi**2

# Plot
plt.plot(x, u_eksakt,'*b', label="Analytisk løsning")
plt.plot(x, u_num, 'r', label="Numerisk løsning")
plt.legend()
plt.xlabel("x")
plt.ylabel("u(x)")
plt.show()