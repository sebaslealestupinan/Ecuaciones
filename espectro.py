"""
espectro.py
Espectro de respuesta del registro sísmico.

Para cada periodo T (o frecuencia fn) y cada amortiguamiento zeta se simula un
oscilador SDOF con el MISMO registro y se guarda solo su respuesta máxima:

    Sd(T, zeta)  = max |x(t)|                  espectro de desplazamiento    [m]
    PSv          = wn * Sd                     pseudo-velocidad              [m/s]
    PSa          = wn^2 * Sd                   pseudo-aceleración            [m/s^2]
    Sv           = max |x'(t)|                 velocidad relativa máxima     [m/s]
    Sa           = max |x'' + ag|              aceleración absoluta máxima   [m/s^2]

Como la masa se cancela en la ecuación, el espectro solo depende de (T, zeta) y del registro.
"""
import numpy as np

from integrador import rk4_sdof
from lector_peer import G


def espectro_respuesta(ag, dt, periodos=None, zetas=(0.02, 0.05, 0.10, 0.20)):
    """
    Devuelve un dict con arreglos de forma (n_zetas, n_periodos).
    `periodos` en segundos (por defecto 0.05-4 s, 120 puntos espaciados logarítmicamente).
    """
    if periodos is None:
        periodos = np.geomspace(0.05, 4.0, 120)
    periodos = np.asarray(periodos, float)
    zetas = np.asarray(zetas, float)

    wn = 2 * np.pi / periodos[None, :]       # (1, nT)
    z = zetas[:, None]                       # (nZ, 1)
    r = rk4_sdof(ag, dt, wn, z, historial=False)

    Sd = r["x_max"]
    return {
        "T": periodos,
        "zetas": zetas,
        "Sd": Sd,
        "PSv": wn * Sd,
        "PSa": wn ** 2 * Sd,
        "Sv": r["v_max"],
        "Sa": r["a_abs_max"],
        "PGA": float(np.max(np.abs(ag))),
        "PSa_g": wn ** 2 * Sd / G,
    }
