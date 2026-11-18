"""
senal_sintetica.py
Genera un acelerograma SINTÉTICO (no es un sismo real) para probar el simulador
mientras no se tiene el .AT2 de PEER. Ruido blanco filtrado (tipo Kanai-Tajimi)
modulado por una envolvente que sube, se mantiene y decae.

Sirve solo para verificar que el código funciona. Los resultados del informe
deben salir del registro real de L'Aquila.
"""
import numpy as np

from lector_peer import G, Registro


def generar_sintetico(dur=40.0, dt=0.005, pga_g=0.30, f_suelo=2.5, zeta_suelo=0.6, semilla=2009):
    rng = np.random.default_rng(semilla)
    n = int(round(dur / dt)) + 1
    t = np.arange(n) * dt

    # Ruido blanco -> filtro de suelo (oscilador de 1 gdl en el dominio de la frecuencia)
    ruido = rng.standard_normal(n)
    freqs = np.fft.rfftfreq(n, dt)
    w = 2 * np.pi * freqs
    wg = 2 * np.pi * f_suelo
    H = (wg ** 2 + 2j * zeta_suelo * wg * w) / (wg ** 2 - w ** 2 + 2j * zeta_suelo * wg * w)
    señal = np.fft.irfft(np.fft.rfft(ruido) * H, n)

    # Envolvente: sube 3 s, se mantiene hasta 12 s, decae exponencialmente
    env = np.where(t < 3, (t / 3) ** 2, np.where(t < 12, 1.0, np.exp(-(t - 12) / 8)))
    ag_g = señal * env
    ag_g *= pga_g / np.max(np.abs(ag_g))

    return Registro(titulo="SEÑAL SINTÉTICA (solo para pruebas, NO es L'Aquila)",
                    t=t, dt=dt, ag_g=ag_g, ag=ag_g * G)
