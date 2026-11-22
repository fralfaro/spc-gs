# Resultados (rejilla 256×512, Truck, semilla 449)

(Movido desde el README. Las tablas se copiaron de `results/metrics_2d.csv` y `results/metrics_3d.csv`.)

## Resultados

Escena Truck, rejilla 256×512, 24 vistas (21 de entrenamiento y 3 de prueba), semilla 449, sin
ruido de medición. Todos los valores se copiaron de `results/metrics_2d.csv` y
`results/metrics_3d.csv`; las tablas y gráficos completos se generan con
`notebooks/03_resultados.ipynb` (figuras en `results/figuras/`). Cada celda es
**PSNR (dB) / SSIM / LPIPS**. Las referencias (`original`, `control`) no usan SPC y no dependen de
la razón de muestreo.

### Reconstrucción 2D (promedio de las 24 vistas vs imagen original)

| Condición | Referencia | 5 % | 10 % | 25 % | 50 % |
|---|---|---|---|---|---|
| control (solo remuestreo) | 27.22 / 0.858 / 0.195 | | | | |
| hadamard · directa | | 20.63 / 0.529 / 0.588 | 21.89 / 0.582 / 0.521 | 23.92 / 0.702 / 0.391 | 25.92 / 0.799 / 0.281 |
| hadamard · tv | | 21.52 / 0.597 / 0.552 | 22.89 / 0.660 / 0.470 | 24.87 / 0.754 / 0.342 | 26.33 / 0.814 / 0.260 |
| random · directa | | 12.65 / 0.245 / 0.650 | 12.93 / 0.197 / 0.652 | 13.87 / 0.170 / 0.648 | 15.82 / 0.202 / 0.629 |
| random · tv | | 20.35 / 0.554 / 0.602 | 21.89 / 0.602 / 0.553 | 24.39 / 0.701 / 0.440 | 26.12 / 0.785 / 0.315 |
| dct · directa | | 21.61 / 0.583 / 0.545 | 22.89 / 0.640 / 0.468 | 24.91 / 0.748 / 0.351 | 26.43 / 0.823 / 0.248 |
| dct · tv | | 21.80 / 0.602 / 0.547 | 23.18 / 0.665 / 0.467 | 25.17 / 0.763 / 0.328 | 26.49 / 0.820 / 0.251 |

![Métricas 2D](results/figuras/metricas_2d.png)

### 3DGS: vistas de prueba vs imágenes originales (métrica principal)

**3000 iteraciones**

| Condición | Referencia | 5 % | 10 % | 25 % | 50 % |
|---|---|---|---|---|---|
| original (baseline) | 17.97 / 0.676 / 0.365 | | | | |
| control (solo remuestreo) | 18.49 / 0.661 / 0.414 | | | | |
| hadamard · directa | | 17.63 / 0.510 / 0.563 | 17.60 / 0.541 / 0.537 | 18.25 / 0.601 / 0.480 | 18.20 / 0.642 / 0.438 |
| hadamard · tv | | 17.88 / 0.553 / 0.566 | 18.15 / 0.586 / 0.523 | 18.28 / 0.627 / 0.470 | 18.36 / 0.650 / 0.438 |
| random · directa | | 12.16 / 0.317 / 0.675 | 12.37 / 0.275 / 0.663 | 12.76 / 0.219 / 0.652 | 13.88 / 0.235 / 0.629 |
| random · tv | | 17.53 / 0.523 / 0.616 | 17.62 / 0.549 / 0.578 | 18.50 / 0.602 / 0.510 | 18.57 / 0.639 / 0.459 |

**7000 iteraciones**

| Condición | Referencia | 5 % | 10 % | 25 % | 50 % |
|---|---|---|---|---|---|
| original (baseline) | 17.72 / 0.662 / 0.355 | | | | |
| control (solo remuestreo) | 18.25 / 0.651 / 0.413 | | | | |
| hadamard · directa | | 17.18 / 0.486 / 0.566 | 17.18 / 0.512 / 0.539 | 17.79 / 0.580 / 0.481 | 18.08 / 0.628 / 0.439 |
| hadamard · tv | | 17.81 / 0.538 / 0.565 | 17.90 / 0.574 / 0.523 | 17.90 / 0.614 / 0.470 | 18.07 / 0.639 / 0.434 |
| random · directa | | 12.13 / 0.264 / 0.667 | 12.21 / 0.213 / 0.667 | 12.49 / 0.172 / 0.662 | 13.41 / 0.183 / 0.647 |
| random · tv | | 17.27 / 0.507 / 0.616 | 17.43 / 0.532 / 0.581 | 18.05 / 0.586 / 0.517 | 18.16 / 0.626 / 0.460 |

![Métricas 3D, prueba, 7000 iteraciones](results/figuras/metricas_3d_it7000.png)

### 3DGS: vistas de entrenamiento vs imágenes originales (complemento)

**3000 iteraciones**

| Condición | Referencia | 5 % | 10 % | 25 % | 50 % |
|---|---|---|---|---|---|
| original (baseline) | 28.81 / 0.914 / 0.208 | | | | |
| control (solo remuestreo) | 25.90 / 0.824 / 0.293 | | | | |
| hadamard · directa | | 20.68 / 0.559 / 0.542 | 21.89 / 0.614 / 0.489 | 23.69 / 0.721 / 0.391 | 25.20 / 0.793 / 0.324 |
| hadamard · tv | | 21.46 / 0.608 / 0.533 | 22.71 / 0.667 / 0.467 | 24.39 / 0.750 / 0.375 | 25.40 / 0.796 / 0.327 |
| random · directa | | 12.80 / 0.349 / 0.673 | 13.12 / 0.291 / 0.661 | 14.18 / 0.249 / 0.642 | 16.40 / 0.286 / 0.610 |
| random · tv | | 20.32 / 0.571 / 0.587 | 21.76 / 0.619 / 0.538 | 23.95 / 0.708 / 0.436 | 25.18 / 0.773 / 0.354 |

**7000 iteraciones**

| Condición | Referencia | 5 % | 10 % | 25 % | 50 % |
|---|---|---|---|---|---|
| original (baseline) | 32.51 / 0.952 / 0.152 | | | | |
| control (solo remuestreo) | 26.63 / 0.840 / 0.268 | | | | |
| hadamard · directa | | 20.63 / 0.548 / 0.554 | 21.89 / 0.603 / 0.498 | 23.86 / 0.717 / 0.391 | 25.66 / 0.802 / 0.312 |
| hadamard · tv | | 21.48 / 0.605 / 0.537 | 22.80 / 0.666 / 0.467 | 24.67 / 0.755 / 0.368 | 25.90 / 0.806 / 0.310 |
| random · directa | | 12.74 / 0.288 / 0.665 | 13.04 / 0.228 / 0.660 | 14.03 / 0.191 / 0.651 | 16.08 / 0.226 / 0.627 |
| random · tv | | 20.32 / 0.567 / 0.591 | 21.83 / 0.617 / 0.542 | 24.19 / 0.710 / 0.437 | 25.73 / 0.783 / 0.341 |

![Métricas 3D, entrenamiento, 7000 iteraciones](results/figuras/metricas_3d_ent_it7000.png)

### Observaciones

1. **2D.** La calidad cae de forma monótona al reducir las mediciones. Con Hadamard ordenado y TV,
   la pérdida respecto del control es de 5.70, 4.33, 2.35 y 0.89 dB al 5, 10, 25 y 50 %. La TV
   mejora a la directa en Hadamard entre 0.41 y 1.00 dB. Hadamard ordenado y DCT quedan a menos
   de 1 dB entre sí.
2. **Aleatorio.** La reconstrucción directa no sirve (12.65–15.82 dB): al medir coeficientes al
   azar, la parte no medida se reparte por toda la imagen como ruido. Con TV, el patrón aleatorio
   se acerca a Hadamard (1.17 dB menos al 5 %, 0.21 dB menos al 50 %).
3. **3DGS, vistas de prueba.** El PSNR casi no distingue condiciones: todas, salvo aleatorio
   directa, quedan entre 17.18 y 18.57 dB, y el control e incluso algunas condiciones SPC superan
   al baseline (ver más abajo qué tan concluyentes son estas diferencias). El SSIM y el LPIPS sí ordenan las
   condiciones como se espera: el baseline tiene el mejor SSIM y el mejor LPIPS, y Hadamard
   (ambos métodos) y aleatorio TV mejoran de forma monótona con la razón en ambas métricas.
4. **3DGS, vistas de entrenamiento.** El modelo 3D hereda casi exactamente la pérdida 2D. Por
   ejemplo, Hadamard TV a 7000 iteraciones obtiene 21.48 dB al 5 % (2D: 21.52) y 25.90 dB al 50 %
   (2D: 26.33). Frente al baseline (32.51 dB), la pérdida es de 5.9 dB solo por la rejilla
   (control) y de 11.0 dB con Hadamard TV al 5 %.
5. **Iteraciones.** Pasar de 3000 a 7000 iteraciones mejora mucho el baseline en las vistas de
   entrenamiento (28.81 → 32.51 dB). En cambio, en las condiciones SPC el cambio es de 0.55 dB
   como máximo, porque ya se ajustaron a sus imágenes degradadas. En las vistas de prueba, el PSNR baja levemente en todas
   las condiciones (p. ej., baseline 17.97 → 17.72 dB), lo que es consistente con sobreajuste a
   pocas vistas.
6. **Recursos (RTX 4060, 8 GB).** Una corrida de 3000 iteraciones tardó 63–90 s y una de 7000,
   146–362 s. La excepción fue `hadamard_directa_10` a 3000 iteraciones, que tardó 1213 s. Sus
   métricas son normales; la demora se atribuye al equipo (probablemente ahorro de energía) y la
   corrida no se volvió a ejecutar. La memoria máxima
   reservada por PyTorch fue de 4.25 GiB (aleatorio directa al 50 %, 7000 iteraciones). Esa
   condición es también la que más gaussianas genera (1.89 M), porque intenta ajustar el ruido
   de la reconstrucción. El máximo leído con `nvidia-smi` durante todo el experimento fue de
   4875 MiB, incluyendo la memoria que usa Windows.

### Qué muestran los renders de prueba

`scripts/diagnostico_prueba.py` muestra, para cada vista de prueba, el render, el error
absoluto, la opacidad acumulada y cuántas de las 21 cámaras de entrenamiento ven el punto 3D de
cada píxel. Ese número se calcula con la profundidad renderizada, sin considerar oclusiones,
así que es una cota superior.

![Comparación de renders de prueba, 7000 iteraciones](results/figuras/comparacion_prueba_it7000_s449_f1.png)

![Diagnóstico del baseline, 7000 iteraciones](results/figuras/diagnostico_prueba_original_it7000.png)

Lo que se verificó (baseline, 7000 iteraciones; `results/diagnostico_prueba_original_it7000.csv`):

- **Hay artefactos de generalización visibles, parecidos en todas las condiciones, incluido el
  baseline:** manchas oscuras y borrosas en el suelo (000131, abajo a la derecha), estelas de
  colores en el fondo izquierdo (000218) y una mancha junto a la puerta (000044).
- **Los píxeles sin gaussianas** (opacidad acumulada < 0.5) son pocos (0.8–2.2 % de la imagen),
  pero tienen el mayor error absoluto medio (0.16–0.24, frente a 0.06–0.11 en toda la imagen).
- **El error es menor donde más cámaras ven el punto**, pero la relación no es monótona. En las
  tres vistas, los píxeles vistos por 11 o más cámaras tienen el menor error (0.053, 0.067 y
  0.089). Sin embargo, los vistos por 6–10 cámaras tienen más error (0.094, 0.131 y 0.172) que
  los vistos por 3–5 (0.078, 0.089 y 0.110). Según el mapa de conteo, esos píxeles
  corresponden sobre todo al fondo (árboles, edificios) y al suelo alejado del camión. El fondo
  además tiene mucha textura fina.

Por lo tanto, la presencia de artefactos flotantes está **verificada visualmente**. La
explicación de que el error de prueba se deba a «zonas poco observadas» es **solo parcialmente
consistente** con los datos y queda como **hipótesis**. La textura del fondo y la falta de
oclusiones en el conteo podrían explicar el resto. También es una hipótesis que estos
artefactos, comunes a todas las condiciones, sean la causa de que el PSNR de prueba apenas
distinga entre ellas.

<!-- VARIABILIDAD -->

### 30000 iteraciones (solo baseline y control)

| Condición | Prueba: PSNR / SSIM / LPIPS | Entrenamiento: PSNR / SSIM / LPIPS | Gaussianas | Tiempo | Memoria máx. reservada |
|---|---|---|---|---|---|
| original (baseline) | 17.20 / 0.619 / 0.367 | 40.23 / 0.980 / 0.080 | 2.49 M | 3495 s | 5.65 GiB |
| control (solo remuestreo) | 18.06 / 0.631 / 0.415 | 27.04 / 0.851 / 0.238 | 0.92 M | 1140 s | 2.22 GiB |

Con más iteraciones, el baseline se ajusta cada vez más a las vistas de entrenamiento (28.81 →
32.51 → 40.23 dB a 3000, 7000 y 30000 iteraciones) y empeora en las de prueba (17.97 → 17.72 →
17.20 dB; SSIM 0.676 → 0.662 → 0.619). Esto es consistente con sobreajuste a 21 vistas. El
control, en cambio, queda limitado por la rejilla en las vistas de entrenamiento (25.90 → 26.63
→ 27.04 dB). Las condiciones SPC no se entrenaron con 30000 iteraciones por tiempo (de 19 a
58 minutos por corrida en estas dos condiciones; entre 5 y 15 horas para las 16). El tiempo del
baseline incluye unos minutos en que la GPU se compartió con `diagnostico_prueba.py`. El máximo
de `nvidia-smi` durante estas corridas fue de 6727 MiB, incluyendo ese script y la memoria que
usa Windows.

### Respuesta a la pregunta

En 2D, medir el 5 % de los coeficientes cuesta unos 6 dB respecto del remuestreo y el 50 %, menos de
1 dB, con Hadamard ordenado o DCT. TV ayuda siempre y es imprescindible con patrones aleatorios.
En 3D, la evaluación en vistas de prueba con 21 vistas es poco sensible en PSNR, y conviene leer
el SSIM y el LPIPS. En las vistas de entrenamiento, 3DGS no corrige la degradación del SPC:
la reproduce con casi la misma magnitud que en 2D.

## Limitaciones

- Se usan solo 21 vistas de entrenamiento (régimen de pocas vistas), así que los valores
  absolutos de 3DGS son menores que los publicados para Truck con todas las vistas. Lo que
  importa es la comparación entre condiciones.
- Una sola escena y una sola semilla: las diferencias pequeñas entre condiciones pueden estar
  dentro de la variabilidad del entrenamiento.
- La evaluación 3D usa 3 vistas de prueba. Con tan pocas vistas, el PSNR de prueba es poco
  sensible a la condición (ver Resultados); por eso se agregan las métricas en vistas de
  entrenamiento, que miden la herencia de la degradación pero no la generalización.
- No se modeló el ruido de medición en los experimentos reportados (el código lo permite).

