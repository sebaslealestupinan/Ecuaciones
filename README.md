# Simulador SDOF ante un sismo real (L'Aquila 2009, PEER NGA-West2)

Sistema masa-resorte-amortiguador de un grado de libertad:

    m x'' + c x' + k x = -m a_g(t),   x(0) = 0, x'(0) = 0

## Estructura

| Archivo | Qué hace (paso del pipeline) |
|---|---|
| `lector_peer.py` | Lee el `.AT2`, extrae Δt y aceleraciones, convierte g → m/s² (1) |
| `sistema.py` | Define (m, fn, ζ), calcula k y c; valores de partida y barridos (2) |
| `integrador.py` | RK4 (principal), Newmark-β y `solve_ivp` (validación) (3) |
| `graficas.py` | a_g(t), x(t), v(t), paramétrico, espectros (4 y 5) |
| `espectro.py` | Espectro de respuesta (extra) |
| `senal_sintetica.py` | Señal de prueba (NO es sismo real) |
| `main.py` | Ejecuta todo y guarda figuras y CSV en `resultados/` |

## Uso

    python main.py                           # prueba con señal sintética
    python main.py RSN_xxx_AQUILA_xxx.AT2    # con el registro real
    python main.py registro.AT2 --relleno 10 # +10 s de ceros (vibración libre final)

Requiere Python 3 con numpy, scipy y matplotlib.

## Valores de partida (propuestos; el documento del proyecto no los fija)

- m = 1000 kg, fn = 1 Hz (T = 1 s), ζ = 5 %.
- Barrido de fn: 0.5, 1, 2, 5 Hz. Barrido de ζ: 2, 5, 10, 20 %. Barrido de m: 500, 1000, 2000 kg.
- Se cambian en `sistema.py`.

## Notas de método

- **RK4 con registro discreto:** a_g se interpola linealmente entre muestras. Cada intervalo
  se subdivide automáticamente para que ωn·h ≤ 0.2.
- **Validación:** RK4 se compara con `solve_ivp` (DOP853, rtol 1e-9) y con Newmark-β.
- **La masa no afecta x(t):** al dividir la ecuación entre m queda
  x'' + 2ζωn x' + ωn² x = -a_g, que solo depende de (ωn, ζ). La masa cambia k, c y las fuerzas.
- **Aceleración absoluta:** x'' + a_g = -(ωn² x + 2ζωn v).
- **Registro real:** se usa una sola componente horizontal; conviene decir en el informe cuál
  (RSN, estación, componente, distancia y V_S30).

## Espectro de respuesta

Ver la explicación en el chat; en el código, `espectro.espectro_respuesta(ag, dt)`.
