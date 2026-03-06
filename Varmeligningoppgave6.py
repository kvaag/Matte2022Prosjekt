
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

plt.close("all")

# -------------------------------------------------------------
# 1) Fysiske parametre og gitter
# -------------------------------------------------------------

# Termisk diffusivitet for metall (aluminium bronze) (22.9 mm²/s = 2.29e-5 m²/s)
alpha = 2.29e-5  # m²/s

# Dimensjoner på tverrsnittet i m
Lx_fysisk = 0.10  # 10 cm
Ly_fysisk = 0.10  # 10 cm

# Temperaturer
T_ovn = 200.0   # °C randbetingelse på alle kanter
T_start = 15.0  # °C initialbetingelse overalt inne

# Antall INDRE punkter i x og y retning
m = 30  # indre punkter i x
n = 30  # indre punkter i y

# Gitter inkludert randpunkter
x = np.linspace(0, Lx_fysisk, m + 2)
y = np.linspace(0, Ly_fysisk, n + 2)

h = x[1] - x[0]  # dx
k = y[1] - y[0]  # dy

x_in = x[1:-1]  # indre punkter (m stk)
y_in = y[1:-1]  # indre punkter (n stk)



# -------------------------------------------------------------
# 2) 2D Laplace-operator L med Kroneckerprodukt
#    L = kron(Lx, Iy) + kron(Ix, Ly)
#    Størrelse: (m*n) × (m*n)
# -------------------------------------------------------------

Lx = (1 / h**2) * (
    np.diag((m - 1) * [1], -1) +
    np.diag(m * [-2], 0) +
    np.diag((m - 1) * [1], 1)
)
Ix = np.eye(m)

Ly = (1 / k**2) * (
    np.diag((n - 1) * [1], -1) +
    np.diag(n * [-2], 0) +
    np.diag((n - 1) * [1], 1)
)
Iy = np.eye(n)

L = np.kron(Lx, Iy) + np.kron(Ix, Ly)

#print(f"Størrelse på L: {L.shape}")
#print(L)

# -------------------------------------------------------------
# 3) Randbetingelser (Dirichlet) alle kanter = T_ovn
#
# Alle fire randfunksjoner konstante = 200°C
# -------------------------------------------------------------

def f1(x):
    """u(x, 0) (Bunnrand)"""
    return T_ovn * np.ones_like(x)

def f2(x):
    """u(x, Ly) (Topprand)"""
    return T_ovn * np.ones_like(x)

def f3(y):
    """u(0, y) (Venstre rand)"""
    return T_ovn * np.ones_like(y)

def f4(y):
    """u(Lx, y) (Høyre rand)"""
    return T_ovn * np.ones_like(y)

# -------------------------------------------------------------
# 4) Bygger F fra randbetingelsene
#
# Randbidraget: naboverdier utenfor det indre området erstattes
# av kjente randverdier og flyttes over til høyresiden.
# -------------------------------------------------------------

Zm_l = np.zeros(m) 
Zm_l[0]  = -1 / h**2   # venstre (x = 0)
Zm_r = np.zeros(m)
Zm_r[-1] = -1 / h**2   # høyre  (x = Lx)
Zn_l = np.zeros(n)
Zn_l[0]  = -1 / k**2   # bunn   (y = 0)
Zn_r = np.zeros(n)
Zn_r[-1] = -1 / k**2   # topp   (y = Ly)

F = (
    np.kron(f1(x_in), Zn_l) +  # y = 0
    np.kron(f2(x_in), Zn_r) +  # y = Ly
    np.kron(Zm_l, f3(y_in)) +  # x = 0
    np.kron(Zm_r, f4(y_in))    # x = Lx
)
print(F)
#print(f"Størrelse på F: {F.shape}")

# -------------------------------------------------------------
# 5) Forlengs Euler for u'(t) = alpha * L * u - alpha * F
#
# Vi ganger med alpha fordi vi må ta hensyn til den termiske diffusiteten(alpha) i varmeligningen som er u_t = alpha(u_xx + u_yy)
# -------------------------------------------------------------

def euler(g, u0, t0, t1, N):
    """
    Løser u' = g(u, t) med forlengs Euler.

    Parametre:
      g  : høyreside g(u, t)
      u0 : initialtilstand (vektor, lengde m*n)
      t0 : starttid
      t1 : sluttid
      N  : antall tidspunkter (inkl. start)

    Returnerer:
      u : løsning, form (N, m*n)
      t : tidsgitter (N,)
    """
    t = np.linspace(t0, t1, N)
    dt = t[1] - t[0]

    u = np.zeros((N, u0.size))
    u[0, :] = u0

    for i in range(N - 1):
        u[i + 1, :] = u[i, :] + dt * g(u[i, :], t[i])

    return u, t


def g(u, t):
    """Høyreside i ODE-systemet: u'(t) = alpha * L * u - alpha * F"""
    return alpha * (L @ u - F)


# -------------------------------------------------------------
# 6) Rutenett for plotting og initialtilstand
# indexing="ij"
# -------------------------------------------------------------

X, Y = np.meshgrid(x_in, y_in, indexing="ij")

# Initialtilstand: materialet starter som 15°C
U0 = T_start * np.ones((m, n))

# Vektorisering: Endrer dimensjonen på U0 fra matrise til en vektor. Rekkefølgen må stemme med L og F
u0 = np.reshape(U0, m * n)

#print(f"Dimensjon på u0: {u0.shape}")

# -------------------------------------------------------------
# 7) Simuler 
# Velger stor N slik at forlengs euler holder seg stabil med mange tidssteg
# -------------------------------------------------------------

total_tid = 90  # sekunder


# Kjør simuleringen

u_løsning, t_vektor = euler(g, u0, 0.0, total_tid, 10000)


# -------------------------------------------------------------
# 8) Finn indeks for midtpunktet og spor temperaturen der
# -------------------------------------------------------------

midtpunkt_i = m // 2
midtpunkt_j = n // 2

# Indeksen i den flate vektoren (linearisert rutenett)
midtpunkt_flat = midtpunkt_i * n + midtpunkt_j

# Temperatur i midten for alle tidssteg
temperatur_midten = u_løsning[:, midtpunkt_flat]

# Finn når midten når 60°C
indeks_60 = np.argmax(temperatur_midten >= 60.0)
if temperatur_midten[indeks_60] >= 60.0:
    tid_60_sekunder = t_vektor[indeks_60]
    print(f"Midten når 60°C ved t = {tid_60_sekunder:.0f} s = {tid_60_sekunder/60:.1f} min")
else:
    tid_60_sekunder = None
    print("Midten nådde ikke 60°C i løpet av simuleringen.")

print(f"Slutttemperatur i midten: {temperatur_midten[-1]:.1f}°C")
print()

# -------------------------------------------------------------
# 9) Varmeplot ved ulike tidspunkter
# -------------------------------------------------------------

# Tidspunkter vi vil plotte (i minutter)
#plot_tider_min = [0, 0.1, 0.2, 0.3, 0.6, 1]
plot_tider_sek = [0,10,20,30,45,60]
plt.figure(figsize=(20, 8))
plt.suptitle("Oppvarming av metallbit i ovn (200°C) — Varmeplot", fontsize=16, fontweight="bold")

for idx, t_sek in enumerate(plot_tider_sek):
    #t_sek = t_min * 60
    # Finn nærmeste tidssteg
    steg_indeks = np.argmin(np.abs(t_vektor - t_sek))

    # Reshape fra vektor til 2D
    Z = np.reshape(u_løsning[steg_indeks, :], (m, n))

    plt.subplot(2, 3, idx + 1)
    plt.imshow(
        Z.T,
        origin="lower",
        aspect="auto",
        extent=[0, Lx_fysisk * 100, 0, Ly_fysisk * 100],
        cmap="RdYlBu_r",
        vmin=T_start,
        vmax=T_ovn,
    )
    plt.colorbar(label="°C")
    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")

    temp_midt = Z[midtpunkt_i, midtpunkt_j]
    plt.title(f"t = {t_sek} sekunder (midt: {temp_midt:.0f}°C)", fontsize=10)

    # Vis midttemperaturen som tekst på plottet
    tekstfarge = "white" if temp_midt < 120 else "black"
    plt.text(
        Lx_fysisk * 100 / 2, Ly_fysisk * 100 / 2,
        f"{temp_midt:.0f}°C",
        ha="center", va="center",
        fontsize=10, fontweight="bold", color=tekstfarge,
        bbox=dict(boxstyle="round,pad=0.2",
                  facecolor="black" if temp_midt < 120 else "white",
                  alpha=0.5),
    )

plt.tight_layout()
plt.savefig("varmeplot_tidspunkter.png", dpi=150, bbox_inches="tight")
plt.show()
print("Lagret: varmeplot_tidspunkter.png")

# -------------------------------------------------------------
# 10) Varmeplot når midten akkurat når 60°C
# -------------------------------------------------------------

if tid_60_sekunder is not None:
    Z_60 = np.reshape(u_løsning[indeks_60, :], (m, n))

    plt.figure(figsize=(8, 5))
    plt.imshow(
        Z_60.T,
        origin="lower",
        aspect="auto",
        extent=[0, Lx_fysisk * 100, 0, Ly_fysisk * 100],
        cmap="RdYlBu_r",
        vmin=T_start,
        vmax=T_ovn,
    )
    plt.colorbar(label="°C")
    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")
    plt.title(f"Varmeplot når midten når 60°C (t = {tid_60_sekunder:.1f} sekunder)",
              fontsize=13, fontweight="bold")
    plt.text(
        Lx_fysisk * 100 / 2, Ly_fysisk * 100 / 2,
        "60°C", ha="center", va="center",
        fontsize=12, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5),
    )
    plt.tight_layout()
    plt.savefig("varmeplot_60grader.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Lagret: varmeplot_60grader.png")

# -------------------------------------------------------------
# 11) 3D-plott av løsningen ved t der midten = 60°C
# -------------------------------------------------------------

if tid_60_sekunder is not None:
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X * 100, Y * 100, Z_60, cmap="RdYlBu_r")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_zlabel("Temperatur (°C)")
    ax.set_title(f"3D-plott ved t = {tid_60_sekunder:.1f} sekunder (midten = 60°C)")

    plt.tight_layout()
    plt.savefig("3d_plott_60grader.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Lagret: 3d_plott_60grader.png")


# ------------------------------------------------------------
# 12) Animasjon av 2D-løsningen u(x,y,t) som fargekart
# ------------------------------------------------------------

# Hvor mange frames vil vi vise?
# Vi tar et hopp på 'stride' tidssteg mellom hver frame for å få en raskere animasjon.
stride = 50
n_frames = 90  # som i originalkoden

# Vi lager en liste med faktiske tidindekser vi vil bruke (sikrer at vi ikke går utenfor)
frame_idx = [i * stride for i in range(n_frames) if i * stride < u_løsning.shape[0]]

# ------------------------------------------------------------
# Forbered "fargeskala" (samme skala i alle frames gir roligere animasjon)
# ------------------------------------------------------------
# Vi bruker de frame-verdiene vi faktisk skal vise når vi bestemmer min/max.
U_frames = np.array([u_løsning[j, :] for j in frame_idx])
vmin = np.min(U_frames)
vmax = np.max(U_frames)

# ------------------------------------------------------------
# Sett opp figur
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

# Første frame som startbilde
Z0 = np.reshape(u_løsning[frame_idx[0], :], (m, n))

im = ax.imshow(
    -Z0.T,                 # .T for å få "riktig" orientering i imshow
    origin="lower",
    cmap="RdYlBu", #endret til RdYlBu for mer spenn i fargekartet
    vmin=-vmax, vmax=-vmin  # symmetrisk skala rundt 0 (nyttig for RdBu)
)

# Akse-etiketter (indre gitterpunkter)
ax.set_title("Tidsutvikling av u(x,y,t) (fargekart)")
ax.set_xlabel("x-indeks (indre punkter)")
ax.set_ylabel("y-indeks (indre punkter)")

# Fargebar gjør det lettere å tolke verdier
plt.colorbar(im, ax=ax, label="u-verdi")

# ------------------------------------------------------------
# Oppdateringsfunksjon (mer minnevennlig enn å lagre 90 bilder i en liste)
# ------------------------------------------------------------
def animate(frame_number):
    j = frame_idx[frame_number]
    Z = np.reshape(u_løsning[j, :], (m, n))
    im.set_data((-Z).T)
    ax.set_title(f"Tidsutvikling av u(x,y,t)  (t ≈ {j} tidssteg)")
    return (im,)

# ------------------------------------------------------------
# Lag animasjonen
# ------------------------------------------------------------
ani = animation.FuncAnimation(
    fig,
    animate,
    frames=len(frame_idx),
    interval=50,
    blit=True
)

HTML(ani.to_jshtml())
plt.show()