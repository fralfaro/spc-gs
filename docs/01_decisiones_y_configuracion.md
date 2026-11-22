# Decisiones de diseño y configuración exacta

(Movido desde el README.)

## Decisiones de diseño

Cuando había dudas se eligió la opción más simple. Estas son las decisiones tomadas:

- **Vistas:** 24 vistas equiespaciadas de las 251 (por nombre). Las de prueba son las posiciones
  4, 12 y 20 de esa selección (000044, 000131 y 000218); las otras 21 entrenan. La
  inicialización usa los puntos 3D de COLMAP, calculados con las 251 vistas; es la misma para
  todas las condiciones.
- **Rejilla:** 256×512 (potencias de 2, necesarias para Hadamard). Su relación de aspecto (2.0)
  difiere levemente de la de las imágenes (1.79); la deformación se deshace al volver al tamaño
  original y su efecto queda incluido en el control.
- **Formato:** las imágenes reconstruidas se guardan en JPEG de calidad 100 sin submuestreo de
  color, para conservar los mismos nombres `.jpg` que espera COLMAP. El control incluye esta
  recodificación, así que su efecto queda separado del efecto del SPC.
- **Patrón aleatorio:** siempre mide el patrón constante. Sin él, el nivel medio de la imagen queda
  indeterminado (el patrón constante no cambia al permutar píxeles y la TV no puede recuperar una
  constante). En una prueba inicial sin esta medición, la TV quedaba en ~6 dB de PSNR.
- **λ de TV:** 0.003, elegido con un barrido λ ∈ {0.001, 0.003, 0.01, 0.03} sobre la vista
  000130 (que no está entre las 24 del experimento), para Hadamard y aleatorio al 5 y 25 %:
  0.003 fue el mejor o quedó a menos de 0.2 dB del mejor en los cuatro casos. Se usa el mismo
  valor en todas las razones.
- **Patrones ±1:** se simulan directamente. Un DMD real solo muestra 0/1; los ±1 se obtienen con
  dos patrones complementarios (medición diferencial).
- **Ruido:** el modelo es simple (Poisson sobre |y|, lectura gaussiana y cuantización uniforme) y
  está apagado en todos los experimentos reportados.
- **Entrenador 3DGS:** `src/train_3dgs.py` es una versión compacta del `simple_trainer` de gsplat,
  con sus hiperparámetros por defecto (tasas de aprendizaje, `DefaultStrategy` de densificación,
  pérdida 0.8·L1 + 0.2·(1 − SSIM), grado SH 3 que aumenta cada 1000 pasos, fondo negro). Cada
  corrida es independiente: la de 7000 iteraciones no continúa la de 3000.
- **Semilla:** 449 en todo (vistas, patrones, ruido, inicialización y orden de entrenamiento). Los
  kernels de gsplat usan sumas atómicas en la GPU, por lo que dos corridas con la misma semilla
  pueden diferir levemente en 3D.

## Configuración exacta del experimento principal

| Parámetro | Valor |
|---|---|
| Escena | Truck (`tandt_db.zip`): 251 imágenes JPEG de **979×546** px en `data/truck/images`; modelo COLMAP PINHOLE de 1957×1091 (las intrínsecas se escalan a 979×546) |
| Vistas | 24 equiespaciadas; prueba: 000044, 000131 y 000218; las otras 21 entrenan |
| Rejilla SPC | **256×512** (alto × ancho) = 131 072 píxeles por canal, el **25 %** de los 534 534 píxeles de la imagen |
| Medición | cada canal RGB por separado; K = round(r · 131 072) = 6 554, 13 107, 32 768 y 65 536 para r = 5, 10, 25 y 50 % |
| Remuestreo | imagen → rejilla: promedio por área; rejilla → 979×546: bicúbica (`align_corners=False`); recorte a [0, 1] |
| Formato | JPEG calidad 100, sin submuestreo de color (4:4:4) |
| Evaluación 2D | a **resolución completa (979×546)**, después de volver al tamaño original; contra la imagen original; promedio de las 24 vistas |
| Evaluación 3D | render a 979×546 contra la imagen original: 3 vistas de prueba (principal), 21 de entrenamiento (complemento) y 230 no usadas para entrenar (extendida) |
| Métricas | PSNR (rango 1, sobre RGB); SSIM (ventana gaussiana 11×11, σ = 1.5, por canal); LPIPS (red VGG, paquete `lpips`) |
| TV | λ = 0.003, 100 iteraciones de FISTA, 20 de Chambolle |
| 3DGS | gsplat 1.5.3, resolución completa (`FACTOR=1`), grado SH 3, 3000 y 7000 iteraciones (30000 solo baseline y control), semilla 449 |
| Ruido | apagado |

### Por qué el control da 27.22 dB y no ~38 dB

Un informe anterior reportaba unos 38.1 dB para el control con la rejilla 512×1024. Ese informe
no está en este repositorio, así que no se pudo comprobar su configuración. Para aislar los
factores se usó `scripts/analisis_control.py`, que mide el control en las mismas 24 vistas
variando un factor a la vez. Los resultados están en `results/analisis_control.csv` (PSNR
promedio en dB):

| Rejilla | Píxeles de la rejilla / imagen | Evaluado a 979×546, JPEG q100 (como el experimento) | Evaluado a 979×546, sin JPEG | Evaluado a la resolución de la rejilla |
|---|---|---|---|---|
| 256×512 | 25 % | **27.22** | 27.24 | 35.21 |
| 512×1024 | 98 % | 38.00 | 38.14 | 34.75 |
| (sin remuestreo, solo JPEG) | 100 % | 55.37 | — | — |

- **El factor principal es la rejilla.** Con 512×1024, la rejilla tiene casi tantos píxeles como
  la imagen y el remuestreo apenas pierde información: el control queda en 38.0–38.1 dB, igual
  que el valor reportado. Con 256×512 se descarta el 75 % de los píxeles y se pierden los
  detalles finos, lo que da 27.22 dB.
- **La resolución de evaluación también influye.** Si el control a 256×512 se evaluara a la
  resolución de la rejilla, daría 35.21 dB, porque así no se mide el detalle que la rejilla no
  puede representar. Aquí siempre se evalúa a resolución completa.
- **El JPEG de calidad 100 no influye:** cuesta entre 0.02 y 0.14 dB.
- **Corrección de un error.** El análisis detectó un error en `remuestrear`
  (`src/make_dataset.py`). Con la rejilla 512×1024, el alto se reduce (546 → 512) pero el ancho
  se amplía (979 → 1024), y la función ampliaba el ancho con promedio por área. Con esa versión,
  el control a 512×1024 daba 30.58 dB. Se corrigió (bicúbica con antialias en el caso mixto) y
  se agregó una prueba. La rejilla 256×512 no estaba afectada: el análisis reproduce
  exactamente los 27.22 dB.

