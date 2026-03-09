import numpy as np
import matplotlib.pyplot as plt

plt.close("all")

#Diskretiserer i rom: x går fra -1 til 1 fordelt i 102 romlige steg med lengde h
m = 100
x = np.linspace(-1, 1, m + 2)
h = x[1] - x[0]


# L ≈ -d^2/dx^2 på indre punkter
L = (1 / h**2) * (
    np.diag((m - 1) * [1], -1) +
    np.diag(m * [-2], 0) +
    np.diag((m - 1) * [1], 1)
)
# c = 1, så vi utelater den fra utregningene
A = L

# Randbetingelser
a = 0.0
b = 2.0

#f(x)=cos(pi*x)
F = np.cos(np.pi * x[1:-1])
F[0]  -= a / h**2
F[-1] -= b / h**2



# Forlengs Euler: x_{n+1} = x_n + dt * g(x_n, t_n)

def euler(g, x0, t0, t1, N):

    t = np.linspace(t0, t1, N)
    dt = t[1] - t[0]
#Sjekker kravet for tids diskretiseringen:
    Sjekk = dt/h**2
    if Sjekk < 1/2:
        print(True)
    else:
        print(False)

    xsol = np.zeros((N, x0.size))
    xsol[0, :] = x0

    for n in range(N - 1):
        xsol[n + 1, :] = xsol[n, :] + dt * g(xsol[n, :], t[n])

    return xsol, t


# u'(t) = A u(t) - F
def g(u, t):
    return A @ u - F


# Initialbetingelse på indre punkter
u0 = 1+ x[1:-1]+5*np.sin(np.pi * x[1:-1])

# Lager mange små tidssteg, fordi ellers kan forlengs euler være for ustabil
u, t = euler(g, u0, 0.0, 1.0, 10000)

print("dt =", t[1] - t[0])
print("Maksverdi ved start:", np.max(u[0, :]))
print("Maksverdi ved slutt:", np.max(u[-1, :]))

# Plotter løsning ved ulike tider 

plt.figure(figsize=(6, 4))
plt.plot(x[1:-1], u[0, :],  label="t = t[0]")
plt.plot(x[1:-1], u[10, :],  label="t = t[10]")
plt.plot(x[1:-1], u[100, :],  label="t = t[100]")
plt.plot(x[1:-1], u[500, :], label="t = t[500]")
plt.plot(x[1:-1], u[1000, :], label="t = t[1000]")
plt.plot(x[1:-1], u[1500, :], label="t = t[1500]")
plt.plot(x[1:-1], u[4000, :], label="t = t[4000]")

plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Utvikling i tid med forlengs Euler (semi-diskret system)")
plt.legend()
plt.show()