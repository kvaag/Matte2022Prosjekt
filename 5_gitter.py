#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 12:36:14 2026

@author: agathe
"""


import numpy as np
import matplotlib.pyplot as plt


m = 3
n = 3

x = np.linspace(-5, 5, m + 2)
y = np.linspace(0, 2, n + 2)
h = x[1] - x[0]   # dx
k = y[1] - y[0]   # dy



X, Y = np.meshgrid(x, y, indexing="xy")

plt.figure(figsize=(5,5))
plt.plot(X, Y, "k.", markersize=6)   # alle punkter
plt.title("2D-gitter (inkl. randpunkter)")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()
