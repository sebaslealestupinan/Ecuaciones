"""
lector_peer.py
Ingesta y preprocesamiento del registro sísmico (paso 1 del pipeline).

Los archivos .AT2 de PEER NGA-West2 tienen este formato:
    línea 1: "PEER NGA STRONG MOTION DATABASE RECORD"
    línea 2: evento, fecha, estación, componente
    línea 3: "ACCELERATION TIME SERIES IN UNITS OF G"
    línea 4: "NPTS=  4000, DT=   .0050 SEC"
    resto  : aceleraciones en g, 5 valores por línea

El Δt es constante y viene en la línea 4; no hay columna de tiempo,
así que el vector t se reconstruye como t_i = i * Δt.
"""
import re
from dataclasses import dataclass

import numpy as np

G = 9.80665  # m/s^2 por cada g


@dataclass
class Registro:
    titulo: str
    t: np.ndarray        # [s]
    dt: float            # [s]
    ag_g: np.ndarray     # aceleración del terreno en unidades de g
    ag: np.ndarray       # aceleración del terreno en m/s^2

    @property
    def n(self):
        return len(self.ag)

    @property
    def duracion(self):
        return self.dt * (self.n - 1)

    @property
    def pga(self):
        """Aceleración pico del terreno en m/s^2."""
        return float(np.max(np.abs(self.ag)))

    def resumen(self):
        return (f"{self.titulo}\n"
                f"  puntos = {self.n}, dt = {self.dt:g} s, duración = {self.duracion:.2f} s\n"
                f"  PGA = {self.pga:.3f} m/s^2 ({self.pga / G:.3f} g)")


def leer_at2(ruta):
    """Lee un archivo .AT2 de PEER y devuelve un Registro (aceleración en m/s^2)."""
    with open(ruta, "r", errors="ignore") as f:
        lineas = f.read().splitlines()

    # Se busca la línea "NPTS= ..., DT= ..." (normalmente la 4, pero no se asume)
    idx = next((i for i, ln in enumerate(lineas[:12]) if "NPTS" in ln.upper()), None)
    if idx is None:
        raise ValueError("No se encontró la línea 'NPTS=..., DT=...' en el encabezado del .AT2")

    m_npts = re.search(r"NPTS\s*=\s*(\d+)", lineas[idx], re.I)
    m_dt = re.search(r"DT\s*=\s*([0-9.eE+\-]+)", lineas[idx], re.I)
    if not (m_npts and m_dt):
        raise ValueError(f"No se pudo interpretar la línea de encabezado: {lineas[idx]!r}")
    npts, dt = int(m_npts.group(1)), float(m_dt.group(1))

    valores = []
    for ln in lineas[idx + 1:]:
        valores.extend(float(x) for x in ln.replace(",", " ").split())
    ag_g = np.array(valores, dtype=float)

    if len(ag_g) < npts:
        raise ValueError(f"El archivo declara NPTS={npts} pero solo tiene {len(ag_g)} datos")
    ag_g = ag_g[:npts]

    titulo = " | ".join(s.strip() for s in lineas[:idx - 1] if s.strip()) if idx > 1 else "registro PEER"
    t = np.arange(npts) * dt
    return Registro(titulo=titulo, t=t, dt=dt, ag_g=ag_g, ag=ag_g * G)


def leer_texto_simple(ruta, unidades="g"):
    """
    Alternativa para archivos de dos columnas (t, a) separados por espacios, comas o tabs.
    `unidades` = "g" o "m/s2".
    """
    datos = np.loadtxt(ruta, delimiter=None if "," not in open(ruta).read(2000) else ",")
    t, a = datos[:, 0], datos[:, 1]
    dts = np.diff(t)
    if not np.allclose(dts, dts[0], rtol=1e-3):
        raise ValueError("El paso de tiempo no es constante; remuestrear antes de usar el simulador")
    ag_g = a if unidades == "g" else a / G
    return Registro(titulo=str(ruta), t=t - t[0], dt=float(dts[0]), ag_g=ag_g, ag=ag_g * G)


def rellenar_ceros(registro, segundos):
    """
    Agrega ceros al final del registro para observar la vibración libre
    posterior al sismo (útil si el máximo puede ocurrir después de que termina el registro).
    """
    extra = int(round(segundos / registro.dt))
    ag = np.concatenate([registro.ag, np.zeros(extra)])
    return Registro(titulo=registro.titulo, t=np.arange(len(ag)) * registro.dt,
                    dt=registro.dt, ag_g=ag / G, ag=ag)
