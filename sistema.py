"""
sistema.py
Configuración del sistema SDOF (paso 2 del pipeline).

Se define con (m, fn, zeta) y se calculan (k, c):
    wn = 2*pi*fn,   k = m*wn^2,   c = 2*zeta*m*wn
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SistemaSDOF:
    m: float      # masa [kg]
    fn: float     # frecuencia natural [Hz]
    zeta: float   # razón de amortiguamiento [-]

    @property
    def wn(self):
        return 2.0 * math.pi * self.fn

    @property
    def T(self):
        return 1.0 / self.fn

    @property
    def k(self):
        return self.m * self.wn ** 2

    @property
    def c(self):
        return 2.0 * self.zeta * self.m * self.wn

    def __str__(self):
        return (f"SDOF(m={self.m:g} kg, fn={self.fn:g} Hz, T={self.T:.3f} s, zeta={self.zeta:.1%}) "
                f"-> k={self.k:.4g} N/m, c={self.c:.4g} N*s/m")


# ---------------------------------------------------------------------------
# VALORES DE PARTIDA PROPUESTOS (el markdown del proyecto no los fija)
# ---------------------------------------------------------------------------
# m = 1000 kg  : masa típica de un modelo de laboratorio / piso de una estructura pequeña.
#                Como x(t) no depende de m (ver README), el valor es solo ilustrativo.
# fn = 1.0 Hz  : T = 1 s, periodo típico de un edificio de unos 8-10 pisos.
# zeta = 5 %   : valor estándar para estructuras de concreto/acero en ingeniería sísmica.
M_BASE = 1000.0
FN_BASE = 1.0
ZETA_BASE = 0.05

# Barridos del análisis paramétrico
MASAS = [500.0, 1000.0, 2000.0]            # kg
FRECUENCIAS = [0.5, 1.0, 2.0, 5.0]         # Hz  (T = 2, 1, 0.5, 0.2 s)
AMORTIGUAMIENTOS = [0.02, 0.05, 0.10, 0.20]
