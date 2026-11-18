"""
graficas.py
Visualización (paso 4) y gráficas del análisis paramétrico (paso 5) y del espectro.

Criterios de diseño: líneas finas, rejilla tenue, un color por entidad en orden fijo,
magnitudes ordenadas (frecuencia) en una sola gama azul de claro a oscuro, y sin ejes dobles.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lector_peer import G

# Paleta (slots categóricos 1-4 en orden fijo) y tintas
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURF = "#fcfcfb"
# Secuencial de un solo tono (azul) para magnitudes ordenadas, de claro a oscuro
SEQ = ["#9ec5f4", "#5a9be6", "#2a78d6", "#154a91", "#0b2a54"]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.4, "legend.frameon": False, "legend.fontsize": 9,
    "font.family": "DejaVu Sans",
})


def _marcar_pico(ax, t, y, color, unidad, factor=1.0, fmt=".2f"):
    i = int(np.argmax(np.abs(y)))
    ax.plot(t[i], y[i] * factor, "o", ms=6, color=color, mec=SURF, mew=1.5, zorder=5)
    # Etiqueta fija en la esquina superior derecha del panel (nunca se sale del eje)
    ax.text(0.99, 0.95, f"pico {abs(y[i]) * factor:{fmt}} {unidad}  (t = {t[i]:.1f} s)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK2)


def graficar_respuesta(reg, res, sistema, ruta):
    """a_g(t), x(t) y v(t) en tres paneles con el eje de tiempo compartido."""
    t = reg.t
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    ax[0].plot(t, reg.ag_g, color=CAT[3])
    ax[0].set_ylabel("a_g  [g]")
    ax[0].set_title("Aceleración del terreno")
    _marcar_pico(ax[0], t, reg.ag_g, CAT[3], "g", fmt=".3f")

    x_mm = res["x"][:, 0] * 1000
    ax[1].plot(t, x_mm, color=CAT[0])
    ax[1].set_ylabel("x  [mm]")
    ax[1].set_title("Desplazamiento relativo de la masa")
    _marcar_pico(ax[1], t, x_mm, CAT[0], "mm", fmt=".1f")

    v_cm = res["v"][:, 0] * 100
    ax[2].plot(t, v_cm, color=CAT[1])
    ax[2].set_ylabel("v  [cm/s]")
    ax[2].set_title("Velocidad relativa de la masa")
    ax[2].set_xlabel("Tiempo [s]")
    _marcar_pico(ax[2], t, v_cm, CAT[1], "cm/s", fmt=".1f")

    for a in ax:
        a.axhline(0, color=INK2, lw=0.6)
    fig.suptitle(f"{sistema}", x=0.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def graficar_validacion(reg, x_rk4, x_new, x_sci, ruta):
    """Comparación de los tres métodos y error respecto a la referencia de alta precisión."""
    t = reg.t
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(t, x_sci * 1000, color=INK2, lw=2.6, alpha=0.5, label="solve_ivp (referencia)")
    ax[0].plot(t, x_rk4 * 1000, color=CAT[0], lw=1.2, label="RK4")
    ax[0].plot(t, x_new * 1000, color=CAT[1], lw=1.2, ls="--", label="Newmark-β")
    ax[0].set_ylabel("x  [mm]")
    ax[0].set_title("Los tres métodos coinciden")
    ax[0].legend(loc="upper right", ncol=3)

    pico = np.max(np.abs(x_sci))
    ax[1].plot(t, (x_rk4 - x_sci) / pico * 100, color=CAT[0], label="RK4")
    ax[1].plot(t, (x_new - x_sci) / pico * 100, color=CAT[1], ls="--", label="Newmark-β")
    ax[1].set_ylabel("error  [% del pico]")
    ax[1].set_xlabel("Tiempo [s]")
    ax[1].set_title("Diferencia respecto a la referencia")
    ax[1].legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def graficar_parametrico_fn(reg, resultados, ruta):
    """
    x(t) para varias frecuencias naturales (zeta fija). Un tono azul, de claro (fn baja)
    a oscuro (fn alta), con etiqueta directa en cada curva.
    """
    t = reg.t
    fig, ax = plt.subplots(figsize=(9, 5))
    for (fn, x), col in zip(resultados, SEQ):
        ax.plot(t, x * 1000, color=col, lw=1.2, label=f"fn = {fn:g} Hz (T = {1/fn:g} s)")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("x  [mm]")
    ax.set_title("Efecto de la frecuencia natural en el desplazamiento")
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def graficar_parametrico_zeta(reg, resultados, ruta):
    """x(t) para varios amortiguamientos (fn fija). Colores categóricos en orden fijo."""
    t = reg.t
    fig, ax = plt.subplots(figsize=(9, 5))
    for (z, x), col in zip(resultados, CAT):
        ax.plot(t, x * 1000, color=col, lw=1.2, label=f"ζ = {z:.0%}")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("x  [mm]")
    ax.set_title("Efecto del amortiguamiento en el desplazamiento")
    ax.legend(loc="upper right", ncol=4)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def graficar_masa(masas, x_max_mm, f_max_kn, ruta):
    """
    La masa no cambia el desplazamiento (si fn y zeta son fijos), pero sí la fuerza en el resorte.
    Dos paneles separados (cada uno con su escala), nunca un eje doble.
    """
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    etiquetas = [f"{m:g}" for m in masas]
    ax[0].bar(etiquetas, x_max_mm, color=CAT[0], width=0.55)
    ax[0].set_title("Desplazamiento máximo")
    ax[0].set_ylabel("|x|max  [mm]")
    ax[0].set_xlabel("Masa [kg]")
    ax[0].set_ylim(0, max(x_max_mm) * 1.25)
    for i, v in enumerate(x_max_mm):
        ax[0].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=INK2)

    ax[1].bar(etiquetas, f_max_kn, color=CAT[1], width=0.55)
    ax[1].set_title("Fuerza máxima en el resorte")
    ax[1].set_ylabel("k·|x|max  [kN]")
    ax[1].set_xlabel("Masa [kg]")
    ax[1].set_ylim(0, max(f_max_kn) * 1.25)
    for i, v in enumerate(f_max_kn):
        ax[1].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=INK2)
    for a in ax:
        a.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def graficar_mapa_picos(fns, zetas, matriz_mm, ruta):
    """Mapa de calor del |x|max para la malla (fn, zeta). Un solo tono, claro a oscuro."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("azul", ["#e8f1fc", "#2a78d6", "#0b2a54"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(matriz_mm, cmap=cmap, aspect="auto", origin="lower")
    ax.set_xticks(range(len(zetas)), [f"{z:.0%}" for z in zetas])
    ax.set_yticks(range(len(fns)), [f"{f:g}" for f in fns])
    ax.set_xlabel("Amortiguamiento ζ")
    ax.set_ylabel("Frecuencia natural fn [Hz]")
    ax.set_title("Desplazamiento máximo |x|max [mm]")
    ax.grid(False)
    vmax = matriz_mm.max()
    for i in range(len(fns)):
        for j in range(len(zetas)):
            val = matriz_mm[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if val > 0.55 * vmax else INK)
    fig.colorbar(im, ax=ax, label="mm")
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def graficar_espectros(esp, ruta):
    """Espectros de respuesta: PSa [g], PSv [cm/s] y Sd [cm] contra el periodo T."""
    T = esp["T"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.4))
    datos = [
        (esp["PSa"] / G, "Pseudo-aceleración  PSa  [g]", "Espectro de pseudo-aceleración"),
        (esp["PSv"] * 100, "Pseudo-velocidad  PSv  [cm/s]", "Espectro de pseudo-velocidad"),
        (esp["Sd"] * 100, "Desplazamiento  Sd  [cm]", "Espectro de desplazamiento"),
    ]
    for a, (Y, ylab, tit) in zip(ax, datos):
        for z, y, col in zip(esp["zetas"], Y, CAT):
            a.plot(T, y, color=col, label=f"ζ = {z:.0%}")
        a.set_xscale("log")
        a.set_xlabel("Periodo T [s]")
        a.set_ylabel(ylab)
        a.set_title(tit)
        a.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 4], ["0.05", "0.1", "0.2", "0.5", "1", "2", "4"])
        a.set_xlim(T[0], T[-1])
        a.set_ylim(bottom=0)
    ax[0].axhline(esp["PGA"] / G, color=INK2, lw=0.9, ls=":")
    ax[0].annotate("PGA", (T[-1], esp["PGA"] / G), xytext=(-4, 4), textcoords="offset points",
                   ha="right", fontsize=8.5, color=INK2)
    ax[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)
