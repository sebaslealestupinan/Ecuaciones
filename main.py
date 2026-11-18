"""
main.py  -  Simulador SDOF ante un registro sísmico real

Uso:
    python main.py                          # prueba con señal sintética
    python main.py ruta/al/registro.AT2     # con el registro real de PEER
    python main.py registro.AT2 --relleno 10   # agrega 10 s de ceros para ver la vibración libre final

Genera figuras PNG y tablas CSV en la carpeta  resultados/.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import graficas as gr
from espectro import espectro_respuesta
from integrador import newmark_sdof, rk4_sdof, scipy_sdof
from lector_peer import G, leer_at2, rellenar_ceros
from senal_sintetica import generar_sintetico
from sistema import (AMORTIGUAMIENTOS, FN_BASE, FRECUENCIAS, M_BASE, MASAS, ZETA_BASE,
                     SistemaSDOF)


def main():
    ap = argparse.ArgumentParser(description="Simulador SDOF ante un sismo (PEER NGA-West2)")
    ap.add_argument("at2", nargs="?", help="archivo .AT2 de PEER (si se omite, usa una señal sintética)")
    ap.add_argument("--salida", default="resultados", help="carpeta de salida")
    ap.add_argument("--relleno", type=float, default=0.0,
                    help="segundos de ceros al final del registro (vibración libre)")
    args = ap.parse_args()
    os.makedirs(args.salida, exist_ok=True)
    out = lambda nombre: os.path.join(args.salida, nombre)

    # 1) Ingesta y preprocesamiento ------------------------------------------------------
    reg = leer_at2(args.at2) if args.at2 else generar_sintetico()
    if args.relleno > 0:
        reg = rellenar_ceros(reg, args.relleno)
    print("== 1. Registro ==")
    print(reg.resumen())

    # 2) Sistema base ---------------------------------------------------------------------
    base = SistemaSDOF(M_BASE, FN_BASE, ZETA_BASE)
    print("\n== 2. Sistema base ==")
    print(base)

    # 3) Integración (RK4) ----------------------------------------------------------------
    print("\n== 3. Integración RK4 ==")
    res = rk4_sdof(reg.ag, reg.dt, base.wn, base.zeta)
    x, v = res["x"][:, 0], res["v"][:, 0]
    print(f"  |x|max = {np.max(np.abs(x)) * 1000:.2f} mm   |v|max = {np.max(np.abs(v)) * 100:.2f} cm/s")
    gr.graficar_respuesta(reg, res, base, out("01_respuesta_base.png"))

    # Validación cruzada: RK4 vs Newmark vs solve_ivp -------------------------------------
    nw = newmark_sdof(reg.ag, reg.dt, base.wn, base.zeta)
    sc = scipy_sdof(reg.ag, reg.dt, base.wn, base.zeta)
    pico = np.max(np.abs(sc["x"]))
    err_rk4 = np.max(np.abs(x - sc["x"])) / pico
    err_nw = np.max(np.abs(nw["x"] - sc["x"])) / pico
    print(f"  Validación (error máx. relativo al pico de solve_ivp): RK4 = {err_rk4:.2e}, Newmark = {err_nw:.2e}")
    gr.graficar_validacion(reg, x, nw["x"], sc["x"], out("02_validacion_metodos.png"))

    # 5) Análisis paramétrico -------------------------------------------------------------
    print("\n== 5. Análisis paramétrico ==")
    filas = []

    # a) Variar fn (zeta y m fijos)
    res_fn = []
    for fn in FRECUENCIAS:
        s = SistemaSDOF(M_BASE, fn, ZETA_BASE)
        r = rk4_sdof(reg.ag, reg.dt, s.wn, s.zeta)
        res_fn.append((fn, r["x"][:, 0]))
        filas.append(["fn", s.m, fn, s.zeta, r["x_max"][0] * 1000, r["v_max"][0] * 100])
    gr.graficar_parametrico_fn(reg, res_fn, out("03_parametrico_fn.png"))

    # b) Variar zeta (fn y m fijos)
    res_z = []
    for z in AMORTIGUAMIENTOS:
        s = SistemaSDOF(M_BASE, FN_BASE, z)
        r = rk4_sdof(reg.ag, reg.dt, s.wn, s.zeta)
        res_z.append((z, r["x"][:, 0]))
        filas.append(["zeta", s.m, FN_BASE, z, r["x_max"][0] * 1000, r["v_max"][0] * 100])
    gr.graficar_parametrico_zeta(reg, res_z, out("04_parametrico_zeta.png"))

    # c) Variar m (fn y zeta fijos): x no cambia, la fuerza sí
    x_m, f_m = [], []
    for m in MASAS:
        s = SistemaSDOF(m, FN_BASE, ZETA_BASE)
        r = rk4_sdof(reg.ag, reg.dt, s.wn, s.zeta, historial=False)
        xm = r["x_max"][0]
        x_m.append(xm * 1000)
        f_m.append(s.k * xm / 1000)
        filas.append(["m", m, FN_BASE, ZETA_BASE, xm * 1000, r["v_max"][0] * 100])
    gr.graficar_masa(MASAS, x_m, f_m, out("05_parametrico_masa.png"))

    # d) Mapa (fn x zeta) de |x|max
    fns_mapa = [0.25, 0.5, 1.0, 2.0, 4.0]
    matriz = np.zeros((len(fns_mapa), len(AMORTIGUAMIENTOS)))
    for i, fn in enumerate(fns_mapa):
        for j, z in enumerate(AMORTIGUAMIENTOS):
            s = SistemaSDOF(M_BASE, fn, z)
            matriz[i, j] = rk4_sdof(reg.ag, reg.dt, s.wn, s.zeta, historial=False)["x_max"][0] * 1000
    gr.graficar_mapa_picos(fns_mapa, AMORTIGUAMIENTOS, matriz, out("06_mapa_fn_zeta.png"))

    with open(out("tabla_parametrico.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["parametro_variado", "m_kg", "fn_Hz", "zeta", "x_max_mm", "v_max_cm_s"])
        for fila in filas:
            w.writerow([fila[0], f"{fila[1]:g}", f"{fila[2]:g}", f"{fila[3]:g}",
                        f"{fila[4]:.4f}", f"{fila[5]:.4f}"])
    print(f"  {len(filas)} casos guardados en tabla_parametrico.csv")
    for f in filas:
        print(f"  {f[0]:>5s}: m={f[1]:>6g} kg  fn={f[2]:>4g} Hz  zeta={f[3]:.0%}  ->  "
              f"|x|max={f[4]:8.2f} mm   |v|max={f[5]:7.2f} cm/s")

    # 6) Espectro de respuesta ------------------------------------------------------------
    print("\n== 6. Espectro de respuesta ==")
    esp = espectro_respuesta(reg.ag, reg.dt, zetas=AMORTIGUAMIENTOS)
    gr.graficar_espectros(esp, out("07_espectros_respuesta.png"))
    j = int(np.argmin(np.abs(np.array(AMORTIGUAMIENTOS) - 0.05)))
    k = int(np.argmax(esp["PSa"][j]))
    print(f"  Para zeta = 5 %: PSa máx = {esp['PSa'][j, k] / G:.3f} g en T = {esp['T'][k]:.2f} s "
          f"(PGA = {esp['PGA'] / G:.3f} g, amplificación = {esp['PSa'][j, k] / esp['PGA']:.2f})")
    with open(out("espectro_respuesta.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["T_s"] + [f"Sd_m_z{z:.0%}" for z in esp["zetas"]]
                   + [f"PSv_m_s_z{z:.0%}" for z in esp["zetas"]]
                   + [f"PSa_g_z{z:.0%}" for z in esp["zetas"]])
        for i, T in enumerate(esp["T"]):
            w.writerow([f"{T:.4f}"] + [f"{esp['Sd'][q, i]:.6f}" for q in range(len(esp["zetas"]))]
                       + [f"{esp['PSv'][q, i]:.6f}" for q in range(len(esp["zetas"]))]
                       + [f"{esp['PSa_g'][q, i]:.6f}" for q in range(len(esp["zetas"]))])
    print(f"\nListo. Figuras y tablas en: {os.path.abspath(args.salida)}")


if __name__ == "__main__":
    main()
