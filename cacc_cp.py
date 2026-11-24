#--------------------Controllore CACC a spaziatura costante (CP)
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import sumolib
import traci
from scipy.integrate import solve_ivp   # solve_ivp: integratore Runge-Kutta 4/5 a passo variabile, come ode45 di MATLAB

# ----------------------------File -------------------------------------------------------------------------
cartella = Path(__file__).parent / "scenario"
configurazione = cartella / "road.configurazione.sumocfg"
plotone = cartella / "road.plotone.xml"

# ------------------------Parametri----------------------------------
N = 30                     # numero di follower (N+1 veicoli con il leader)
L = 5.0                    # lunghezza veicolo
d = 20.0                   # distanza desiderata Δp̄
Kp, Kv = 1.0, 2.0          # K_Δp, K_Δv
lam = 1.5                  # polo del filtro ρ
MACRO = True              # False: a = b = 0 (solo microscopico) | True: +informazione macroscopica
a, b = (0.5, 0.5) if MACRO else (0.0, 0.0)   # pesi dell'informazione macroscopica (Tabella I)
g_p, g_v = 0.5, 0.5        # γ_Δp, γ_Δv, pesi della funzione ψ
UPS = 0.9                  # Υ ∈ (0, 1), parametro della dimostrazione (Tabella I dell'articolo)
U_MAX = 4.0                # |u| ≤ 4 m/s², saturazione dell'accelerazione
V_MAX = 36.0               # 0 ≤ v ≤ 36 m/s
T_FINE = 60.0              # durata della simulazione
x0 = N * (d + L) + 50.0    # posizione iniziale del muso del leader

GUI = True                 # False = senza grafica, più veloce (con passo 0.001 s la GUI rallenta parecchio)
SCHERMO = None             # (larghezza, altezza) dello schermo in pixel, es. (1920, 1080); None = la rileva da sola
SCOPE = True               # la figura dei risultati si disegna durante la simulazione, come lo scope di Simulink
AGGIORNA_SCOPE = 0.1       # ogni quanti secondi simulati ridisegnare la figura
RTOL, ATOL = 1e-8, 1e-10   # tolleranze dell'integratore (quelle di default di ode45 sono 1e-3 e 1e-6)

#--------------condizioni iniziali vicine all'equilibrio-------------------------------
# si parte vicino all'equilibrio: con errori grandi i comandi arrivano alla saturazione
# e il teorema dell'articolo non garantisce più la string stability

rng = np.random.default_rng(1)  # seme fisso: i risultati sono sempre uguali
gap_iniziali = d + rng.uniform(-2.0, 2.0, N + 1)          # gap iniziali tra 18 e 22 m
vel_iniziali = 14.0 + rng.uniform(-1.0, 1.0, N + 1)       # velocità iniziali tra 13 e 15 m/s


#----------------velocità desiderata del leader (fasi 1, 2 e 4)--------------------------------
def v_desiderata(t):
    if 10.0 <= t < 20.0:
        return 25.0
    if 20.0 <= t < 35.0:
        return 20.0
    if 45.0 <= t:
        return 25.0
    return 14.0
# 0<= t < 10 :v=14 m/s
# 10<= t < 20 :v=25 m/s
# 20<= t < 35 :v=20 m/s
# 35<= t < 45 :v=14 m/s
# 45<= t :v=25 m/s

#----------------disturbo sul leader (fasi 3 e 4), che nessuno conosce------------------------------
def disturbo(t, t_passo):
    """Il tratto lo sceglie t_passo (inizio del passo), il seno usa il tempo vero t.
    I cambi di fase (25, 30, 35 s) cadono sempre all'inizio di un passo: così l'integratore non li 'vede' a metà passo."""
    if 25.0 <= t_passo < 30.0:
        return 4.0                      # fase 3: impulso di 4 m/s² per 5 s
    if 35.0 <= t_passo:
        return 2.0 * np.sin(1.0 * t)    # fase 4: sinusoide, ampiezza 2, 1 rad/s
    return 0.0

#------------------funzione ψ per il filtro macroscopico rho--------------------------------
def psi(valori, gamma):             # valori: errori di posizione o di velocità dei veicoli davanti
    """ψ = γ · sign(media) · deviazione standard"""
    valori = np.asarray(valori)
    return gamma * np.sign(valori.mean()) * valori.std()


# ------------Modello continuo del plotone: veicoli + controllore CP + filtro ρ ------------------------------
# stato y = [p0..pN, v0..vN, ρ0..ρN] = posizione del muso, velocità e filtro macroscopico di ogni veicolo
n = N + 1
def plotone_continuo(t, y, v_bar, t_passo, u_registrati=None):
    """Derivate dello stato: ṗ = v, v̇ = u (+ disturbo sul leader), ρ̇ = −λρ + aψ_Δp + bψ_Δv.
    u, ψ e ρ cambiano in modo continuo anche dentro il passo di SUMO.
    Se si passa un array u_registrati, ci scrive dentro l'ingresso applicato uᵢ di ogni veicolo (serve per il grafico)."""
    p, v, r = y[:n], y[n:2 * n], y[2 * n:]

    # errori di ogni coppia: Δp̃ⱼ = d - gapⱼ (0 per il leader), Δvⱼ = vⱼ - vⱼ₋₁ (v0 - v̄ per il leader)
    dp_tilde = np.zeros(n)
    dp_tilde[1:] = d - (p[:-1] - p[1:] - L)
    dv = np.empty(n)
    dv[0] = v[0] - v_bar
    dv[1:] = v[1:] - v[:-1]

    acc = np.empty(n)                    # accelerazione reale di ogni veicolo
    dr = np.empty(n)                     # derivata del filtro di ogni veicolo
    u_trasmesso = 0.0                    # u₋₁ = 0 (leader virtuale)
    for i in range(n):
        # informazione macroscopica: veicoli 0..i-1 (apice i-1 nell'articolo); per i = 0 vale 0
        if i >= 1:
            psi_p = psi(dp_tilde[:i], g_p)
            psi_v = psi(dv[:i], g_v)
        else:
            psi_p = psi_v = 0.0

        # legge Meso(CP):  uᵢ = uᵢ₋₁ - (K_Δp + K_Δv)·Δvᵢ - (1 + K_Δp·K_Δv)·Δp̃ᵢ - ρᵢ
        u = u_trasmesso - (Kp + Kv) * dv[i] - (1 + Kp * Kv) * dp_tilde[i] - r[i]
        u = max(-U_MAX, min(U_MAX, u))               # saturazione
        u_trasmesso = u                              # inviato via V2V al veicolo i+1 (senza il disturbo)
        if u_registrati is not None:
            u_registrati[i] = u                      # ingresso applicato (dopo la saturazione)

        dr[i] = -lam * r[i] + a * psi_p + b * psi_v  # filtro macroscopico: ρ̇ᵢ = -λρᵢ + aψ_Δp + bψ_Δv
        acc[i] = u + (disturbo(t, t_passo) if i == 0 else 0.0)   # per il leader c'è anche il disturbo, che nessuno conosce

    acc[(v >= V_MAX) & (acc > 0)] = 0.0              # 0 ≤ v ≤ Vmax: oltre i limiti la velocità non cambia
    acc[(v <= 0.0) & (acc < 0)] = 0.0
    return np.concatenate([v, acc, dr])


# ------------------Analisi in continuo della coppia, come nell'articolo (Teorema 2, eq. 79-89) ---------------------------
# stato della coppia χ̃ = (Δp̃, Δv, ρ). Senza saturazione u_{i-1} si cancella e resta un sistema lineare
A_c = np.array([[0.0,            1.0,          0.0],
                [-(1 + Kp * Kv), -(Kp + Kv),  -1.0],
                [0.0,            0.0,         -lam]])
autovalori = np.linalg.eigvals(A_c)
# costanti della funzione di Lyapunov W_cp dell'articolo
alfa_min = 0.5                                       # (81)  ½·λmin(P_cp)
alfa_max = 0.5 * (1 + Kp**2)                         # (82)  ½·λmax(P_cp)
alfa_Q = min(Kv, Kp * (1 + Kv * Kp), lam)            # (84)  λmin(Q_cp)
d_m = a * g_p + b * g_v                              # (86)  d = a·γ_Δp + b·γ_Δv
gamma_tilde = np.sqrt(alfa_max / alfa_min) * d_m / (alfa_Q * UPS)   # (89)
print("teoria (continuo): autovalori della coppia (Δp̃, Δv, ρ) =", np.round(autovalori, 4),
      f"  leader: −(Kp + Kv) = {-(Kp + Kv)}")
print(f"                   γ̃ = √({alfa_max}/{alfa_min})·{d_m}/({alfa_Q}·{UPS}) = {gamma_tilde:.4f}"
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
binario = sumolib.checkBinary("sumo-gui" if GUI else "sumo")   # sumo-gui con la grafica, sumo senza
traci.start([binario, "-c", str(configurazione),
             "--route-files", str(plotone),         # tipo di veicolo e percorso del plotone
             "--insertion-checks", "none",               # i veicoli entrano nelle posizioni scelte, senza i controlli di SUMO
             "--collision.action", "warn"])              # SUMO segnala gli urti senza togliere i veicoli
             #+ (["--window-size", f"{META},{ALTEZZA}", "--window-pos", f"{META},0"] if GUI else []))   # SUMO nella metà destra
dt = traci.simulation.getDeltaT()                           # passo di SUMO (0.001 s nella configurazione)

# ------------Plotone con condizioni iniziali vicine all'equilibrio ---------------------------
lista_veicoli = [f"v{i}" for i in range(N + 1)]      # v0 è il leader, N+1 veicoli in totale
pos_iniziale = x0
for i, vid in enumerate(lista_veicoli):
    if i > 0:
        pos_iniziale -= gap_iniziali[i] + L      # posizione del muso: si tolgono il gap e la lunghezza del veicolo
    traci.vehicle.add(vid, "r", typeID="cav", depart="now",
                      departPos=str(pos_iniziale), departSpeed=str(vel_iniziali[i]))
    traci.vehicle.setSpeedMode(vid, 0)           # controlli di SUMO spenti: la velocità la decide lo script
    traci.vehicle.setColor(vid, (200, 40, 40, 255) if i == 0 else (30, 90, 180, 255))   # basta una volta sola

rho = np.zeros(n)        # stato del filtro macroscopico ρᵢ, inizialmente tutti a 0 (esiste solo in Python, lo porta avanti l'integratore)

#-----------------storico per i grafici------------------------------------------------
tempi = []
storico_gap = {vid: [] for vid in lista_veicoli[1:]}      # gap pᵢ₋₁ - pᵢ - L di ogni coppia (non c'è per il leader)
storico_dv = {vid: [] for vid in lista_veicoli}           # vᵢ₋₁ - vᵢ (per v0: v̄ - v0)
storico_rho = {vid: [] for vid in lista_veicoli}
storico_u = {vid: [] for vid in lista_veicoli}            # ingresso applicato uᵢ (accelerazione comandata, senza disturbo)
storico_w0 = []                                           # disturbo sul leader (non lo conosce nessuno), solo per il grafico

# ------------------figura dei risultati (gap, velocità relativa, ρ e ingressi), creata prima del ciclo----------
# con SCOPE = True si aggiorna durante la simulazione, alla fine viene completata e salvata
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
ax3.set_ylabel("ρᵢ")
ax4.set_ylabel("uᵢ [m/s²]")
ax4.set_ylim(-U_MAX - 0.5, U_MAX + 0.5)   # asse fisso sulla saturazione ±U_MAX
ax4.set_xlabel("t [s]")
ax4.set_xlim(0, T_FINE)      # asse dei tempi da 0 a 60 s, senza margini
ax1.set_title(f"CP ———— a = {a}, b = {b}  (chiaro = coppia (0,1), scuro = coppia ({N-1},{N}))",
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
    #-----------------nella GUI la camera segue il veicolo 5------------------------------------------
    if GUI and not seguito and "v5" in traci.vehicle.getIDList():
        traci.gui.trackVehicle("View #0", "v5")
        traci.gui.setZoom("View #0", 1900)
        seguito = True
    #-----------------controllo urti----------------------------------------------------------
    urtati = traci.simulation.getCollidingVehiclesIDList() # ID dei veicoli che si sono urtati
    if urtati:
        print(f"URTO a t={t:.3f} s tra {urtati}: simulazione interrotta")
        break
    #-------------------a ogni passo posizioni e velocità lette da SUMO--------------------
    pos = np.array([traci.vehicle.getLanePosition(vid) for vid in lista_veicoli])  # posizione del muso di ogni veicolo
    vel = np.array([traci.vehicle.getSpeed(vid) for vid in lista_veicoli])         # velocità di ogni veicolo
    v_bar = v_desiderata(t)
    y0 = np.concatenate([pos, vel, rho])         # stato in t (serve anche per l'integrazione sotto)
    u_ora = np.empty(n)
    plotone_continuo(t, y0, v_bar, t, u_ora)     # ingressi applicati in t, solo per il grafico

    #-------------------storico per i grafici (stato in t)------------------------------------------
    tempi.append(t)
    storico_w0.append(disturbo(t, t))
    for i, vid in enumerate(lista_veicoli):
        storico_rho[vid].append(rho[i])
        storico_u[vid].append(u_ora[i])
        storico_dv[vid].append(v_bar - vel[0] if i == 0 else vel[i - 1] - vel[i])   # vᵢ₋₁ - vᵢ (v̄ - v0 per il leader)
        if i >= 1:
            storico_gap[vid].append(pos[i - 1] - pos[i] - L)                        # gap reale

    #-------------------scope: ridisegno la figura ogni AGGIORNA_SCOPE secondi---------------------------
    if SCOPE and t >= prossimo_scope:
        aggiorna_figura()
        fig.canvas.draw_idle()
        plt.pause(0.001)                         # lascia a matplotlib il tempo di ridisegnare
        prossimo_scope += AGGIORNA_SCOPE

    #-------------integrazione del modello continuo da t a t + dt, come ode45-------------------------
    # stato di partenza: posizioni e velocità lette da SUMO, filtro ρ dal passo precedente
    # v̄ e il tratto del disturbo cambiano solo a 10, 20, 25, 30, 35, 45 s, multipli del passo: dentro il passo sono fissi
    sol = solve_ivp(plotone_continuo, (t, t + dt), y0, method="RK45", args=(v_bar, t),
                    rtol=RTOL, atol=ATOL)
    if not sol.success:
        raise RuntimeError(f"integratore fallito a t={t}: {sol.message}")
    v_nuova = sol.y[n:2 * n, -1]                 # velocità di ogni veicolo alla fine del passo
    rho = sol.y[2 * n:, -1]                      # filtro alla fine del passo

    #-------------comando a SUMO: la velocità alla fine del passo; la posizione la calcola SUMO (balistico)------
    for i, vid in enumerate(lista_veicoli):
        traci.vehicle.setSpeed(vid, min(V_MAX, max(0.0, v_nuova[i])))   # 0 ≤ v ≤ V_MAX

traci.close()

# -------------------picco dell'errore di distanza max|gapᵢ - d| per fase, veicoli 1, 2 e N (Tabella 3.1)-------------------
n_passi = len(tempi)
gap_arr = np.array([storico_gap[v][:n_passi] for v in lista_veicoli[1:]])
print(f"\na = {a}, b = {b}")
for fase, (t0, t1) in {"fase 1 (0-10 s)": (0, 10), "fase 2 (10-25 s)": (10, 25),
                       "fase 3 (25-35 s)": (25, 35), "fase 4 (35-60 s)": (35, 60)}.items():
    sel = (np.array(tempi) >= t0) & (np.array(tempi) < t1)
    if sel.any():
        err = np.abs(gap_arr[:, sel] - d).max(axis=1)
        print(f"{fase}: max|gap-d|  v1 = {err[0]:.3f} m   v2 = {err[1]:.3f} m   v{N} = {err[-1]:.3f} m")
print(f"teoria (continuo): γ̃ = {gamma_tilde:.4f}")

# ------------------figura completa, salvata in un file---------------------------------------------------
aggiorna_figura()
if SCOPE:
    plt.ioff()                                   # fine dello scope: plt.show() torna a tenere aperta la finestra
file_png = Path(__file__).parent / f"risultati_cp_{'macro' if MACRO else 'micro'}.png"
fig.savefig(file_png, dpi=150)
print(f"grafico salvato in {file_png.name}")
plt.show()