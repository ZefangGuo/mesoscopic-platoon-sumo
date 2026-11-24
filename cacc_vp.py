#--------------------Controllore CACC a spaziatura VARIABILE (VP)
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import sumolib
import traci
from scipy.integrate import solve_ivp   # solve_ivp: integratore Runge-Kutta 4/5 a passo variabile, come ode45 di MATLAB

# Nel CP ρ entrava nella u direttamente come −ρ, quindi agiva sull’accelerazione. Nel VP ρ1 agisce sulla distanza desiderata
#  e solo dopo arriva alla u. In più ψ entra anche direttamente nella u, con i termini −aψp − bψv, che servono
#  a compensare il fatto che il riferimento si muove
# ψ (veicoli davanti, rispetto a d fisso) → ρ2 → ρ1 → distanza desiderata d + ρ1 → e_p → Δvʳ → u
# Con a = b = 0 il VP coincide con il CP
# il CP corregge l’accelerazione con l’informazione macro, il VP corregge la distanza da tenere e il controllore la insegue



# ----------------------------File -------------------------------------------------------------------------
cartella = Path(__file__).parent / "scenario"
configurazione = cartella / "road.configurazione.sumocfg"
plotone = cartella / "road.plotone.xml"

# ------------------------Parametri----------------------------------
N = 30                     # numero di follower (N+1 veicoli con il leader)
L = 5.0                    # lunghezza veicolo
d = 20.0                   # distanza desiderata Δp̄ (quella "di base", senza informazione macroscopica)
Kp, Kv = 1.0, 2.0          # K_Δp, K_Δv
lam1, lam2 = 1.5, 1.5      # λ1, λ2: velocità dei due stati del filtro ρ1, ρ2
MACRO = True               # False: a = b = 0 (solo microscopico) | True: +informazione macroscopica
a, b = (1.0, 0.2) if MACRO else (0.0, 0.0)   # Tabella II (nel CP erano 0.5 e 0.5)
g_p, g_v = 0.5, 0.5        # γ_Δp, γ_Δv, usati nella funzione ψ
UPS = 0.9                  # Υ ∈ (0, 1), parametro della dimostrazione (come nel CP)
U_MAX = 4.0                # |u| ≤ 4 m/s²
V_MAX = 36.0               # 0 ≤ v ≤ 36 m/s
T_FINE = 60.0              # durata della simulazione
x0 = N * (d + L) + 50.0    # posizione iniziale del muso del leader

GUI = True                 # False = senza grafica, più veloce (con passo 0.001 s la GUI rallenta parecchio)
SCHERMO = None             # (larghezza, altezza) dello schermo in pixel, es. (1920, 1080); None = la rileva da sola
SCOPE = True               # la figura dei risultati si disegna durante la simulazione, come lo scope di Simulink
AGGIORNA_SCOPE = 0.1       # ogni quanti secondi simulati ridisegnare la figura
RTOL, ATOL = 1e-8, 1e-10   # tolleranze dell'integratore (quelle di default di ode45 sono 1e-3 e 1e-6)

#--------------condizioni iniziali vicine all'equilibrio-------------------------------
rng = np.random.default_rng(1)
gap_iniziali = d + rng.uniform(-2.0, 2.0, N + 1)
vel_iniziali = 14.0 + rng.uniform(-1.0, 1.0, N + 1)


#----------------velocità desiderata del leader (identica al CP)--------------------------------
def v_desiderata(t):
    if 10.0 <= t < 20.0:
        return 25.0
    if 20.0 <= t < 35.0:
        return 20.0
    if 45.0 <= t:
        return 25.0
    return 14.0


#----------------disturbo sul leader (identico al CP), che nessuno conosce------------------------------
def disturbo(t, t_passo):
    """Il tratto lo sceglie t_passo (inizio del passo), il seno usa il tempo vero t.
    I cambi di fase (25, 30, 35 s) cadono sempre all'inizio di un passo: così l'integratore non li 'vede' a metà passo."""
    if 25.0 <= t_passo < 30.0:
        return 4.0
    if 35.0 <= t_passo:
        return 2.0 * np.sin(1.0 * t)
    return 0.0


#------------------funzione ψ (identica al CP)--------------------------------
def psi(valori, gamma):
    """ψ = γ · sign(media) · deviazione standard"""
    valori = np.asarray(valori)
    return gamma * np.sign(valori.mean()) * valori.std()


# ------------Modello continuo del plotone: veicoli + controllore VP + filtro a due stati ------------------------------
# stato y = [p0..pN, v0..vN, ρ1,0..ρ1,N, ρ2,0..ρ2,N] = posizione del muso, velocità e i due stati del filtro di ogni veicolo
n = N + 1
def plotone_continuo(t, y, v_bar, t_passo, u_registrati=None):
    """Derivate dello stato: ṗ = v, v̇ = u (+ disturbo sul leader), ρ̇1 = −λ1ρ1 + ρ2, ρ̇2 = −λ2ρ2 + aψp + bψv.
    u, ψ e ρ cambiano in modo continuo anche dentro il passo di SUMO.
    Se si passa un array u_registrati, ci scrive dentro l'ingresso applicato uᵢ di ogni veicolo (serve per il grafico)."""
    p, v, r1, r2 = y[:n], y[n:2 * n], y[2 * n:3 * n], y[3 * n:]

    # errori rispetto a d FISSO (servono per ψ, come nel CP): Δp̃ⱼ = d - gapⱼ ,  Δvⱼ = vⱼ - vⱼ₋₁
    dp_tilde = np.zeros(n)
    dp_tilde[1:] = d - (p[:-1] - p[1:] - L)
    dv = np.empty(n)
    dv[0] = v[0] - v_bar
    dv[1:] = v[1:] - v[:-1]

    acc = np.empty(n)                    # accelerazione reale di ogni veicolo
    dr1 = np.empty(n)                    # derivate dei due stati del filtro
    dr2 = np.empty(n)
    u_trasmesso = 0.0
    for i in range(n):
        if i >= 1:
            psi_p = psi(dp_tilde[:i], g_p)
            psi_v = psi(dv[:i], g_v)
        else:
            psi_p = psi_v = 0.0

        # errore di distanza rispetto al riferimento VARIABILE (17):  Δpᵢ - Δpᵢʳ = Δp̃ᵢ + ρ1,ᵢ = (d + ρ1,ᵢ) - gapᵢ
        e_p = dp_tilde[i] + r1[i]
        # velocità relativa di riferimento (68):  Δvᵢʳ = λ1ρ1,ᵢ - ρ2,ᵢ - K_Δp·(Δpᵢ - Δpᵢʳ)
        dv_rif = lam1 * r1[i] - r2[i] - Kp * e_p

        # legge VP (67)
        u = (u_trasmesso - e_p - Kv * (dv[i] - dv_rif)
             + (Kp - lam1) * (lam1 * r1[i] - r2[i]) + lam2 * r2[i]
             - Kp * dv[i] - a * psi_p - b * psi_v)
        u = max(-U_MAX, min(U_MAX, u))
        u_trasmesso = u                              # inviato via V2V al veicolo i+1 (senza il disturbo)
        if u_registrati is not None:
            u_registrati[i] = u                      # ingresso applicato (dopo la saturazione)

        # filtro a due stati (66)
        dr1[i] = -lam1 * r1[i] + r2[i]
        dr2[i] = -lam2 * r2[i] + a * psi_p + b * psi_v
        acc[i] = u + (disturbo(t, t_passo) if i == 0 else 0.0)

    acc[(v >= V_MAX) & (acc > 0)] = 0.0              # 0 ≤ v ≤ Vmax: oltre i limiti la velocità non cambia
    acc[(v <= 0.0) & (acc < 0)] = 0.0
    return np.concatenate([v, acc, dr1, dr2])


# ------------------Analisi in continuo della coppia, come nell'articolo (Teorema 3, eq. 90-98) ---------------------------
# stato della coppia χ̃ = (Δp̃, Δv, ρ1, ρ2). Senza saturazione u_{i-1} si cancella e resta un sistema lineare
c1 = -(1 + Kp * Kv) + Kv * lam1 + (Kp - lam1) * lam1     # coefficiente di ρ1 in u_i − u_{i−1}
c2 = -Kv - (Kp - lam1) + lam2                            # coefficiente di ρ2 in u_i − u_{i−1}
A_c = np.array([[0.0,            1.0,          0.0,   0.0],
                [-(1 + Kp * Kv), -(Kp + Kv),  c1,    c2],
                [0.0,            0.0,         -lam1,  1.0],
                [0.0,            0.0,          0.0,  -lam2]])
autovalori = np.linalg.eigvals(A_c)
# costanti della funzione di Lyapunov W_vp dell'articolo
alfa_min = 0.5                                                    # (92)  ½·λmin(P_vp)
alfa_max = 0.5 * max(1 + Kp**2, 2 + (lam1 - Kp)**2)              # (93)  ½·λmax(P_vp)
q1 = Kp * (1 + Kp * Kv)
q4 = Kp + lam1 + Kv * (lam1 - Kp)**2
alfa_Q = min(q1, Kv, q4, lam2 + Kv)                               # (95)  λmin(Q_vp)
d_m = a * g_p + b * g_v                                           # (86)  d = a·γ_Δp + b·γ_Δv
gamma_tilde = np.sqrt(alfa_max / alfa_min) * d_m / (alfa_Q * UPS)   # (98)
print("teoria (continuo): autovalori della coppia (Δp̃, Δv, ρ1, ρ2) =", np.round(autovalori, 4),
      f"  leader: −(Kp + Kv) = {-(Kp + Kv)}")
print(f"                   γ̃ = √({alfa_max}/{alfa_min})·{d_m:.1f}/({alfa_Q}·{UPS}) = {gamma_tilde:.4f}"
      f"  ({'< 1: string stability (Teorema 1)' if gamma_tilde < 1 else '≥ 1: condizione non verificata'})")


#-----------------------------Finestre: figura a sinistra, SUMO a destra------------------------------------------------
def dimensioni_schermo():
    """Larghezza e altezza dello schermo in pixel (se non si riesce a leggerle: 1920 x 1080)."""
    if SCHERMO is not None:
        return SCHERMO
    if sys.platform == "win32":
        # con lo zoom di Windows (125%, 150%...) i pixel "veri" sono più di quelli che Python vede di default:
        # SUMO e matplotlib usano quelli veri, quindi li uso anche qui
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass
    try:
        import tkinter
        radice = tkinter.Tk()
        radice.withdraw()
        dimensioni = radice.winfo_screenwidth(), radice.winfo_screenheight()
        radice.destroy()
        return dimensioni
    except Exception:
        return 1920, 1080


def posiziona_figura(figura, x, y, larghezza, altezza):
    """Sposta e ridimensiona la finestra della figura (funziona con le finestre Tk e Qt di matplotlib)."""
    try:
        finestra = figura.canvas.manager.window
    except AttributeError:
        return                                       # nessuna finestra (es. salvataggio senza grafica)
    try:
        geometria = f"{larghezza}x{altezza}+{x}+{y}"
        finestra.wm_geometry(geometria)                              # Tk (TkAgg)
        finestra.after(300, lambda: finestra.wm_geometry(geometria))  # di nuovo dopo che la finestra si è aperta (Windows a volte la sposta)
    except AttributeError:
        try:
            finestra.setGeometry(x, y, larghezza, altezza)          # Qt (QtAgg)
        except AttributeError:
            pass


L_SCHERMO, A_SCHERMO = dimensioni_schermo()
META = L_SCHERMO // 2                                # metà dello schermo
ALTEZZA = A_SCHERMO - 100                            # lascio spazio per la barra delle applicazioni e i bordi


#-----------------------------Avvio --------------------------------------------------------------------------
binario = sumolib.checkBinary("sumo-gui" if GUI else "sumo")
traci.start([binario, "-c", str(configurazione),
             "--route-files", str(plotone),
             "--insertion-checks", "none",
             "--collision.action", "warn"])
             #+ (["--window-size", f"{META},{ALTEZZA}", "--window-pos", f"{META},0"] if GUI else []))   # SUMO nella metà destra
dt = traci.simulation.getDeltaT()                    # passo di SUMO (0.001 s nella configurazione)

# ------------Plotone ---------------------------------------------------------------------------
lista_veicoli = [f"v{i}" for i in range(N + 1)]
pos_iniziale = x0
for i, vid in enumerate(lista_veicoli):
    if i > 0:
        pos_iniziale -= gap_iniziali[i] + L
    traci.vehicle.add(vid, "r", typeID="cav", depart="now",
                      departPos=str(pos_iniziale), departSpeed=str(vel_iniziali[i]))
    traci.vehicle.setSpeedMode(vid, 0)
    traci.vehicle.setColor(vid, (200, 40, 40, 255) if i == 0 else (30, 90, 180, 255))   # basta una volta sola

#------------------il filtro ha DUE stati per veicolo, entrambi nulli all'inizio (esistono solo in Python)---------------------
rho1 = np.zeros(n)       # ρ1,ᵢ = ρᵢᴹ: è lui che sposta la distanza desiderata
rho2 = np.zeros(n)       # ρ2,ᵢ: riceve ψ e "spinge" ρ1

#-----------------storico per i grafici------------------------------------------------
tempi = []
storico_gap = {vid: [] for vid in lista_veicoli[1:]}      # gap reale pᵢ₋₁ - pᵢ - L
storico_gap_rif = {vid: [] for vid in lista_veicoli[1:]}  # gap desiderato d + ρ1,ᵢ (varia nel tempo)
storico_dv = {vid: [] for vid in lista_veicoli}           # vᵢ₋₁ - vᵢ (per v0: v̄ - v0)
storico_rho = {vid: [] for vid in lista_veicoli}          # ρᵢᴹ = ρ1,ᵢ
storico_u = {vid: [] for vid in lista_veicoli}            # ingresso applicato uᵢ (accelerazione comandata, senza disturbo)
storico_w0 = []                                           # disturbo sul leader (non lo conosce nessuno), solo per il grafico

# ------------------grafici (come Fig. 5-7 dell'articolo): la figura si crea prima del ciclo----------------------------
# con SCOPE = True si aggiorna durante la simulazione, alla fine viene completata e salvata come prima
colori = plt.cm.colors.LinearSegmentedColormap.from_list("blu", ["#86b6ef", "#0d366b"])
if SCOPE:
    plt.ion()                                    # modalità interattiva: la figura si aggiorna senza bloccare lo script
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
linee_gap, linee_dv, linee_rho, linee_u = [], [], [], []
for i in range(1, N + 1):
    c = colori((i - 1) / max(N - 1, 1))
    linee_gap.append(ax1.plot([], [], color=c, linewidth=1)[0])
    linee_dv.append(ax2.plot([], [], color=c, linewidth=1)[0])
    linee_rho.append(ax3.plot([], [], color=c, linewidth=1)[0])
    linee_u.append(ax4.plot([], [], color=c, linewidth=1)[0])
linea_dv_leader, = ax2.plot([], [], color="#eb6834", linewidth=1.5, label="v̄ - v0 (leader)")
linea_u_leader, = ax4.plot([], [], color="#eb6834", linewidth=1.5, label="u0 (leader)")
linea_w0, = ax4.plot([], [], color="#52514e", linewidth=1.0, linestyle="--",
                     label="w0 disturbo sul leader (non misurato)")
ax2.legend(frameon=False, fontsize=9, loc="upper right")
ax4.legend(frameon=False, fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 0.98), ncol=2)   # sopra il grafico, non copre le linee
ax1.set_ylabel("pᵢ₋₁ - pᵢ - L [m]")
ax2.set_ylabel("vᵢ₋₁ - vᵢ [m/s]")
ax3.set_ylabel("ρᵢᴹ = ρ1,ᵢ [m]")
ax4.set_ylabel("uᵢ [m/s²]")
ax4.set_ylim(-U_MAX - 0.5, U_MAX + 0.5)   # asse fisso sulla saturazione ±U_MAX
ax4.set_xlabel("t [s]")
ax4.set_xlim(0, T_FINE)
ax1.set_title(f"VP ————— a = {a}, b = {b}  (chiaro = coppia (0,1), scuro = coppia ({N-1},{N}))",
              loc="left", fontsize=11)
for ax in (ax1, ax2, ax3, ax4):
    for tf in (10, 25, 35):
        ax.axvline(tf, color="#c3c2b7", linewidth=0.8, linestyle=":")
    ax.grid(color="#e6e5e0", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
posiziona_figura(fig, 0, 0, META, ALTEZZA)       # figura nella metà sinistra dello schermo
salto = max(1, int(round(0.01 / dt)))            # nei grafici un punto ogni 0.01 s (con 0.001 s sarebbero 60 000 punti per linea)


def aggiorna_figura():
    """Mette nelle linee della figura i dati registrati fino ad ora."""
    tg = tempi[::salto]
    for i in range(1, N + 1):
        vid = lista_veicoli[i]
        linee_gap[i - 1].set_data(tg, storico_gap[vid][::salto])
        linee_dv[i - 1].set_data(tg, storico_dv[vid][::salto])
        linee_rho[i - 1].set_data(tg, storico_rho[vid][::salto])
        linee_u[i - 1].set_data(tg, storico_u[vid][::salto])
    linea_dv_leader.set_data(tg, storico_dv["v0"][::salto])
    linea_u_leader.set_data(tg, storico_u["v0"][::salto])
    linea_w0.set_data(tg, storico_w0[::salto])
    for ax in (ax1, ax2, ax3):                   # ax4 resta fisso a ±U_MAX
        ax.relim()
        ax.autoscale_view(scalex=False)          # l'asse y si adatta ai dati, l'asse x resta 0..T_FINE


# ------------------------------Ciclo ------------------------------------------------------------------------------
seguito = False
prossimo_scope = AGGIORNA_SCOPE
while traci.simulation.getTime() < T_FINE:
    traci.simulationStep()
    t = traci.simulation.getTime()

    if GUI and not seguito and "v5" in traci.vehicle.getIDList():   # nella GUI la camera segue il veicolo 5
        traci.gui.trackVehicle("View #0", "v5")
        traci.gui.setZoom("View #0", 1900)
        seguito = True

    urtati = traci.simulation.getCollidingVehiclesIDList()
    if urtati:
        print(f"URTO a t={t:.3f} s tra {urtati}: simulazione interrotta")
        break

    #-------------------a ogni passo posizioni e velocità lette da SUMO--------------------
    pos = np.array([traci.vehicle.getLanePosition(vid) for vid in lista_veicoli])
    vel = np.array([traci.vehicle.getSpeed(vid) for vid in lista_veicoli])
    v_bar = v_desiderata(t)
    y0 = np.concatenate([pos, vel, rho1, rho2])  # stato in t (serve anche per l'integrazione sotto)
    u_ora = np.empty(n)
    plotone_continuo(t, y0, v_bar, t, u_ora)     # ingressi applicati in t, solo per il grafico

    #-------------------storico per i grafici (stato in t)------------------------------------------
    tempi.append(t)
    storico_w0.append(disturbo(t, t))
    for i, vid in enumerate(lista_veicoli):
        storico_rho[vid].append(rho1[i])
        storico_u[vid].append(u_ora[i])
        storico_dv[vid].append(v_bar - vel[0] if i == 0 else vel[i - 1] - vel[i])   # vᵢ₋₁ - vᵢ (v̄ - v0 per il leader)
        if i >= 1:
            storico_gap[vid].append(pos[i - 1] - pos[i] - L)                        # gap reale
            storico_gap_rif[vid].append(d + rho1[i])                                # gap desiderato d + ρ1

    #-------------------scope: ridisegno la figura ogni AGGIORNA_SCOPE secondi---------------------------
    if SCOPE and t >= prossimo_scope:
        aggiorna_figura()
        fig.canvas.draw_idle()
        plt.pause(0.001)                         # lascia a matplotlib il tempo di ridisegnare
        prossimo_scope += AGGIORNA_SCOPE

    #-------------integrazione del modello continuo da t a t + dt, come ode45-------------------------
    # stato di partenza: posizioni e velocità lette da SUMO, filtro (ρ1, ρ2) dal passo precedente
    # v̄ e il tratto del disturbo cambiano solo a 10, 20, 25, 30, 35, 45 s, multipli del passo: dentro il passo sono fissi
    sol = solve_ivp(plotone_continuo, (t, t + dt), y0, method="RK45", args=(v_bar, t),
                    rtol=RTOL, atol=ATOL)
    if not sol.success:
        raise RuntimeError(f"integratore fallito a t={t}: {sol.message}")
    v_nuova = sol.y[n:2 * n, -1]                 # velocità di ogni veicolo alla fine del passo
    rho1 = sol.y[2 * n:3 * n, -1]                # filtro alla fine del passo
    rho2 = sol.y[3 * n:, -1]

    #-------------comando a SUMO: la velocità alla fine del passo; la posizione la calcola SUMO (balistico)------
    for i, vid in enumerate(lista_veicoli):
        traci.vehicle.setSpeed(vid, min(V_MAX, max(0.0, v_nuova[i])))

traci.close()

# -------------------print picco dell'errore di distanza max|gapᵢ - d| per fase ----------------------------------------
n_passi = len(tempi)
gap_arr = np.array([storico_gap[v][:n_passi] for v in lista_veicoli[1:]])
rif_arr = np.array([storico_gap_rif[v][:n_passi] for v in lista_veicoli[1:]])
print(f"\nVP  a = {a}, b = {b}")
for fase, (t0, t1) in {"fase 1 (0-10 s)": (0, 10), "fase 2 (10-25 s)": (10, 25),
                       "fase 3 (25-35 s)": (25, 35), "fase 4 (35-60 s)": (35, 60)}.items():
    sel = (np.array(tempi) >= t0) & (np.array(tempi) < t1)
    if sel.any():
        err = np.abs(gap_arr[:, sel] - d).max(axis=1)              # rispetto a d fisso (confrontabile col CP)
        err_rif = np.abs(gap_arr[:, sel] - rif_arr[:, sel]).max(axis=1)  # rispetto al proprio riferimento d + ρ1
        print(f"{fase}: max|gap-d|  v1 = {err[0]:.3f}  v2 = {err[1]:.3f}  v{N} = {err[-1]:.3f} m"
              f"   |   max|gap-(d+ρ1)|  v2 = {err_rif[1]:.3f}  v{N} = {err_rif[-1]:.3f} m")
rho_arr = np.array([storico_rho[v][:n_passi] for v in lista_veicoli])
print(f"max|ρᴹ| su tutto il plotone = {np.abs(rho_arr).max():.3f} m")
print(f"teoria (continuo): γ̃ = {gamma_tilde:.4f}")

# ------------------grafici: completo la figura con tutti i dati e la salvo---------------------------------------------------
aggiorna_figura()
if SCOPE:
    plt.ioff()                                   # fine dello scope: plt.show() torna a tenere aperta la finestra
file_png = Path(__file__).parent / f"risultati_vp_{'macro' if MACRO else 'micro'}.png"
fig.savefig(file_png, dpi=150)
print(f"grafico salvato in {file_png.name}")
plt.show()