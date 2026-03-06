#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 15:29:12 2026

@author: agathe
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

# Antall INDRE punkter i x- og y-retning
m = 100 #x-rettning
n = 100 #y-rettning

# Gitter (inkl. randpunkter)
x = np.linspace(-5, 5, m + 2)
y = np.linspace(0, 2, n + 2)

dx = x[1] - x[0] #steglengde i x
dy = y[1] - y[0] #steglengder i y rettning
print("Størrelse på steg:", dy,dx)

Lx = (1 / dx**2) * (
    np.diag((m - 1) * [1], -1) +
    np.diag(m * [-2], 0) +
    np.diag((m - 1) * [1], 1))
Ix = np.eye(m) #identitets matrise med m rader og kolonner

Ly = (1 / dy**2) * (
    np.diag((n - 1) * [1], -1) +
    np.diag(n * [-2], 0) +
    np.diag((n - 1) * [1], 1))
Iy = np.eye(n)#identitets matrise med n rader og kolonner
L = np.kron(Lx, Iy) + np.kron(Ix, Ly)


Zm_l = np.zeros(m)     # venstre kolonne 
Zm_l[0] = -1 / (dx**2) 
#Fyller første element i en tom vektor med -1/(dx**2)


Zm_r = np.zeros(m)     # høyre kolonne 
Zm_r[-1] = -1 / (dx**2)
#Fyller siste element i en tom vektor med -1/(dx**2)


Zn_l = np.zeros(n)     # nederste rad
Zn_l[0] = -1 / (dy**2)
#Fyller første element i en tom vektor med -1/(dy**2)


Zn_r = np.zeros(n)     # øverste rad 
Zn_r[-1] = -1 / (dy**2)
#Fyller siste element i en tom vektor med -1/(dy**2)

def f1(x):
    """u(x,0) (bunnrand)"""
    return 0 * x

def f2(x):
    """u(x,2) (topprand)"""
    return np.sin(np.pi * x)

def f3(y):
    """u(-5,y) (venstre rand)"""
    return np.sin(2*np.pi*y)

def f4(y):
    """u(5,y) (høyre rand)"""
    return np.sin(2*np.pi*y)


x_in = x[1:-1]  # indre x-punkter (m stk)
y_in = y[1:-1]  # indre y-punkter (n stk)

F = (np.kron(f1(x_in), Zn_l) +   # bunnrand y=0
    np.kron(f2(x_in), Zn_r) +   # topprand y=2
    np.kron(Zm_l, f3(y_in)) +   # venstre rand x=-5
    np.kron(Zm_r, f4(y_in)))     # høyre rand x=5

X, Y = np.meshgrid(x_in, y_in, indexing="ij")

def f_source(x, y):
    """Kildeledd f(x,y) i -Δu = f."""
    return 0 * x * y

# Funksjonsverdier på rutenettet (m x n matrise)
Z = f_source(X, Y)

# Vektoriser matrise
G = np.reshape(Z, m * n)

# Legg volumleddet til høyresiden (som allerede inneholder randbidrag)
F = F + G 

print("Størrelse på G:", G.shape)
print("Størrelse på F:", F.shape)


u = la.solve(L, F)
U = np.reshape(u, (m, n))

fig, ax = plt.subplots(
    subplot_kw={"projection": "3d"},
    figsize=(10, 8))

ax.plot_surface(X, Y, U, cmap="PuBu")

# Samme romlige skala som gitteret
ax.set_xlim(x[1], x[-2])
ax.set_ylim(y[1], y[-2])

# Z-skala basert på løsningen (gir stabil visning)
ax.set_zlim(np.min(U), np.max(U))

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("u(x,y)")
ax.set_title("Numerisk løsning av 2D Poisson-problem")

plt.show()

