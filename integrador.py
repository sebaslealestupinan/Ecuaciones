"""
integrador.py
Integración numérica de la EDO (paso 3 del pipeline).

Sistema de primer orden (con m cancelada):
    x' = v
    v' = -wn^2 * x - 2*zeta*wn * v - ag(t)

Métodos incluidos:
  - rk4_sdof     : Runge-Kutta de 4º orden (método principal del proyecto)
  - newmark_sdof : Newmark-beta (aceleración promedio), para validación cruzada
  - scipy_sdof   : scipy.integrate.solve_ivp (DOP853), referencia de alta precisión

Sobre RK4 y el registro discreto:
  RK4 evalúa la derivada en t, t + h/2 y t + h, pero ag(t) solo se conoce en los
  instantes muestreados. Se interpola LINEALMENTE entre muestras (hipótesis estándar
  para registros sísmicos). Si wn*dt es grande, cada intervalo del registro se
  subdivide en n_sub pasos para mantener la precisión de RK4.

rk4_sdof está vectorizado: wn y zeta pueden ser arreglos (se combinan por broadcasting),
así el mismo código resuelve un solo sistema o cientos a la vez (espectro de respuesta).
"""
import math

import numpy as np
from scipy.integrate import solve_ivp


def _n_sub_auto(wn_max, dt, objetivo=0.2):
    """Subpasos para que wn*h <= objetivo dentro de cada intervalo del registro."""
    return max(1, math.ceil(wn_max * dt / objetivo))


def rk4_sdof(ag, dt, wn, zeta, n_sub=None, historial=True):
    """
    Integra x'' + 2*zeta*wn*x' + wn^2*x = -ag(t) con RK4, desde el reposo.

    Parámetros
    ----------
    ag : (N,) aceleración del terreno [m/s^2]
    dt : paso del registro [s]
    wn, zeta : escalares o arreglos (se aplica broadcasting)
    n_sub : subpasos por intervalo (None = automático)
    historial : True  -> devuelve x, v, a_rel, a_abs con forma (N,) + forma_sistemas
                False -> devuelve solo picos (más rápido y sin memoria extra)
    """
    ag = np.asarray(ag, float)
    wn, zeta = np.broadcast_arrays(np.atleast_1d(np.asarray(wn, float)),
                                   np.atleast_1d(np.asarray(zeta, float)))
    forma = wn.shape
    N = len(ag)
    if n_sub is None:
        n_sub = _n_sub_auto(float(wn.max()), dt)
    h = dt / n_sub

    w2 = wn ** 2
    zw2 = 2.0 * zeta * wn

    x = np.zeros(forma)
    v = np.zeros(forma)

    if historial:
        X = np.zeros((N,) + forma)
        V = np.zeros((N,) + forma)
    pico_x = np.zeros(forma)
    pico_v = np.zeros(forma)
    pico_aabs = np.zeros(forma)

    def deriv_v(xx, vv, a):
        return -w2 * xx - zw2 * vv - a

    for i in range(N - 1):
        a0, a1 = ag[i], ag[i + 1]
        da = a1 - a0
        for j in range(n_sub):
            s = j / n_sub
            aa = a0 + da * s
            am = a0 + da * (s + 0.5 / n_sub)
            ab = a0 + da * (s + 1.0 / n_sub)

            k1x = v
            k1v = deriv_v(x, v, aa)
            k2x = v + 0.5 * h * k1v
            k2v = deriv_v(x + 0.5 * h * k1x, v + 0.5 * h * k1v, am)
            k3x = v + 0.5 * h * k2v
            k3v = deriv_v(x + 0.5 * h * k2x, v + 0.5 * h * k2v, am)
            k4x = v + h * k3v
            k4v = deriv_v(x + h * k3x, v + h * k3v, ab)

            x = x + (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
            v = v + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)

            np.maximum(pico_x, np.abs(x), out=pico_x)
            np.maximum(pico_v, np.abs(v), out=pico_v)
            # aceleración absoluta = x'' + ag = -wn^2 x - 2 zeta wn v
            np.maximum(pico_aabs, np.abs(w2 * x + zw2 * v), out=pico_aabs)

        if historial:
            X[i + 1] = x
            V[i + 1] = v

    if not historial:
        return {"x_max": pico_x, "v_max": pico_v, "a_abs_max": pico_aabs}

    a_abs = -(w2 * X + zw2 * V)           # aceleración absoluta
    a_rel = a_abs - ag.reshape((N,) + (1,) * len(forma))  # aceleración relativa
    return {"x": X, "v": V, "a_rel": a_rel, "a_abs": a_abs,
            "x_max": pico_x, "v_max": pico_v, "a_abs_max": pico_aabs}


def newmark_sdof(ag, dt, wn, zeta, gamma=0.5, beta=0.25):
    """Newmark-beta (aceleración promedio constante: gamma=1/2, beta=1/4). Un solo sistema."""
    ag = np.asarray(ag, float)
    N = len(ag)
    k, c = wn ** 2, 2.0 * zeta * wn     # con m = 1
    p = -ag
    x = np.zeros(N)
    v = np.zeros(N)
    a = np.zeros(N)
    a[0] = p[0] - c * v[0] - k * x[0]

    k_hat = k + gamma / (beta * dt) * c + 1.0 / (beta * dt ** 2)
    A = 1.0 / (beta * dt) + gamma / beta * c
    B = 1.0 / (2 * beta) + dt * (gamma / (2 * beta) - 1.0) * c

    for i in range(N - 1):
        dp_hat = (p[i + 1] - p[i]) + A * v[i] + B * a[i]
        dx = dp_hat / k_hat
        dv = gamma / (beta * dt) * dx - gamma / beta * v[i] + dt * (1 - gamma / (2 * beta)) * a[i]
        da = dx / (beta * dt ** 2) - v[i] / (beta * dt) - a[i] / (2 * beta)
        x[i + 1] = x[i] + dx
        v[i + 1] = v[i] + dv
        a[i + 1] = a[i] + da
    return {"x": x, "v": v, "a_rel": a}


def scipy_sdof(ag, dt, wn, zeta):
    """Referencia de alta precisión: solve_ivp (DOP853) con ag interpolada linealmente."""
    ag = np.asarray(ag, float)
    t = np.arange(len(ag)) * dt

    def f(tt, y):
        a = np.interp(tt, t, ag)
        return [y[1], -wn ** 2 * y[0] - 2 * zeta * wn * y[1] - a]

    sol = solve_ivp(f, (t[0], t[-1]), [0.0, 0.0], t_eval=t, method="DOP853",
                    rtol=1e-9, atol=1e-12, max_step=dt)
    return {"x": sol.y[0], "v": sol.y[1]}
