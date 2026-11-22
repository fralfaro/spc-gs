# Objetivo, medición y flujo del experimento

Repositorio base: `spc-gs`. De `inf449-proyecto` solo se trae lo que falta (modelo de ruido DMD y orden cake-cutting).
Escena única: `data/truck` (Tanks and Temples), 24 vistas equiespaciadas (21 de entrenamiento, 3 de prueba), semilla 449.

## 1. Objetivo

Medir **cuánta calidad se pierde al adquirir las vistas con una cámara de un píxel (SPC) simulada**, primero en 2D y luego
en la reconstrucción 3D con Gaussian Splatting, al reducir las mediciones al 5, 10, 25 y 50 % de los píxeles de la rejilla.

Pregunta: ¿cómo cambian la calidad 2D y la calidad 3D según el **patrón** de medición, el **método** de reconstrucción y la
**rejilla**, y cuánto de esa pérdida llega al modelo 3D?

## 2. Qué se mide y contra qué

| Qué | Se compara | Métricas |
|---|---|---|
| Reconstrucción 2D | imagen reconstruida (a 979×546) vs imagen **original** | PSNR, SSIM, LPIPS |
| Control | solo remuestreo a la rejilla y vuelta, sin SPC. Es el techo del 2D. | igual |
| 3DGS, prueba (principal) | render de 3 vistas de prueba vs imágenes **originales** | igual |
| 3DGS, entrenamiento (complemento) | render de las 21 vistas de entrenamiento vs originales | igual |
| Baseline 3DGS | entrenar con las imágenes originales. Es el techo del 3D. | igual |

Regla: nunca se evalúa contra imágenes degradadas, y siempre con la misma función (`src/metrics.py`) y a resolución completa.

## 3. Variables del experimento

| Variable | Valores | Estado |
|---|---|---|
| Patrón | Hadamard por secuencia, aleatorio (con DC), DCT (solo 2D). Candidato: Hadamard cake-cutting | **Por decidir** (paso 2) |
| Método | directa (Φᵀy) y TV-FISTA (λ = 0.003, 100 it.) | Fijo |
| Razón | 5, 10, 25, 50 % | Fijo |
| Rejilla | 256×512 y 512×1024 | 512×1024 aún sin 3D |
| Ruido | apagado (principal); modelo DMD de `inf449` como variante | **Por decidir** (paso 3) |
| Iteraciones 3DGS | 3000 y 7000 (30000 solo baseline y control) | Hecho en 256×512 |

## 4. Flujo

```
imagen original (979×546)
  → remuestreo a la rejilla (promedio por área)
  → medir y = Φx por canal RGB (razón r)
  → reconstruir 2D: directa o TV
  → volver a 979×546 y guardar JPEG q100 con los mismos nombres
  → misma carpeta sparse/0 (cámaras COLMAP originales)
  → entrenar 3DGS con 21 vistas (gsplat)
  → render de 3 vistas de prueba vs originales → PSNR / SSIM / LPIPS
```

Limitación declarada: las poses COLMAP vienen de las imágenes originales, así que el 3D es una **cota optimista**.

## 5. Pasos

| # | Paso | Entregable | Estado |
|---|---|---|---|
| 1 | Entender y fijar los métodos SPC (patrones, directa, TV) y su efecto | Esta sección y `src/patterns.py` | Hecho, falta decidir |
| 2 | Comparar órdenes de Hadamard (secuencia de `spc-gs` vs cake-cutting de `inf449`) en 2D, Truck, 512×1024 | `results/orden_hadamard*.csv` | **Hecho** (ver §7) |
| 3 | Ruido: apagado en lo principal y una variante con el modelo DMD ({0,1} + DC, Poisson, lectura, ADC) | `src/spc.py::medir_dmd`, `scripts/estudio_ruido_dmd.py` | Código y tests listos; corrida 2D pendiente (GPU ocupada) |
| 4 | Tabla 2D única: patrones × métodos × razones × rejillas (256×512 y 512×1024), con control y original | `results/tabla_2d.csv` | Parcial (existe `metrics_2d.csv`) |
| 5 | Validar el baseline 3DGS contra el paper (Truck, 7k: 23.5 / 0.840 / 0.209) y explicar la diferencia con 17.7 dB | `configs/truck_baseline_validacion.yaml` (251 vistas) | Config lista; corre tras el paso 6 |
| 6 | 3DGS en 512×1024 para las condiciones elegidas | `configs/truck_512x1024_3d.yaml` | **En ejecución** (WSL, ~1.2 h) |
| 7 | Tabla 2D → 3D (cuánto de la pérdida 2D llega al 3D) y figuras | `notebooks/03_resultados` | Parcial |
| 8 | Medir tiempos con un mismo cronómetro (solo reconstrucción, por vista) | columna `s_por_vista` | Pendiente |
| 9 | Limpieza: README corto, resultados a `docs/`, `data/truck` fuera de git | Repositorio ordenado | **Hecho** (sin commit) |

## 6. Diferencias con `inf449-proyecto` (por qué no se mezclan tal cual)

- `hadamard_cc` ordena por el número de regiones (s_y+1)(s_x+1); `spc-gs` ordena por u/h + v/w. **No son el mismo patrón.**
- `inf449` simula DMD {0,1} con ruido y parada temprana de la TV; `spc-gs` usa ±1 sin ruido y 100 iteraciones fijas.
- El aleatorio de `spc-gs` mide siempre el DC; en `inf449` no se confirmó.
- Con rejilla 512×1024, ambos coinciden en control (38.0 dB) y aleatorio directa. Hadamard difiere entre 0.3 y 2.3 dB.

## 7. Resultado del paso 2: orden de Hadamard (Truck, 512×1024, 24 vistas, PSNR dB)

Se agregó el patrón `hadamard_cc` (`src/patterns.py`) y el script `scripts/comparar_ordenes_hadamard.py`.

| Patrón · método | 5 % | 10 % | 25 % | 50 % |
|---|---|---|---|---|
| secuencia · directa | 23.17 | 25.05 | 28.48 | 33.12 |
| cake-cutting · directa | 22.81 | 24.74 | 28.46 | 32.89 |
| secuencia · TV | 24.54 | 26.72 | 30.54 | 34.39 |
| cake-cutting · TV | 23.94 | 26.06 | 30.00 | 33.96 |

- **El orden por secuencia es mejor o igual**: gana 0.0 a 0.6 dB. La diferencia es mayor a razones bajas y desaparece en directa al 25 %.
- **No es la causa de la brecha con `inf449`.** Con las mismas 24 vistas de `inf449` (`orden_hadamard_vistas_inf449.csv`),
  cake-cutting TV da 24.00, 26.14, 30.08 y 34.05 dB, frente a 24.11, 26.13, 29.55 y 32.10 en `inf449`.
  A 5 y 10 % coinciden. A 25 y 50 % `inf449` queda 0.5 y 2 dB abajo.
- **El ruido tampoco la explica**: con 1e9 fotones, lectura 1e-5 y ADC de 16 bits (`..._ruido.csv`) el PSNR cambia como máximo 0.12 dB.
- **Los empates del orden tampoco** (otro criterio de desempate dio los mismos números).
- **Sin explicar:** la brecha a razones altas. Queda como hipótesis que venga del modelo DMD {0,1} con medición DC de `inf449`
  (el modelo de `spc-gs` usa ±1 directamente). Verificarlo requiere leer el operador de `inf449`.
- Las 24 vistas de cada proyecto son distintas (`inf449` incluye 000011, 000022…; `spc-gs` 000012, 000023…), pero eso casi no cambia el PSNR.
- Las 5 fallas de `tests/test_cli.py` en Windows son de permisos para crear enlaces simbólicos, no del cambio.
