# Predicción de demanda en panadería artesanal con aprendizaje automático

Código y datos del **Trabajo Fin de Máster** *«Sistema de predicción de demanda basado en
aprendizaje automático para la reducción del desperdicio alimentario en panadería artesanal»* —
Máster Universitario en Inteligencia Artificial, Universidad Internacional de La Rioja (UNIR).

**Autor:** Sebastián Guzmán Uribe

## Descripción

El proyecto desarrolla y valida empíricamente una metodología de predicción de demanda diaria
por categoría de producto para panaderías artesanales, basada en aprendizaje automático y
variables exógenas públicas (meteorología, calendario de festivos y eventos locales), con el
objetivo de reducir el desperdicio alimentario. Se comparan modelos clásicos de series temporales
(SARIMA, Prophet) con métodos de *gradient boosting* (XGBoost, LightGBM) frente a *baselines*
naïve, mediante *backtesting* de un paso adelante con origen rodante, y se estima el impacto en
desperdicio mediante simulación de la política de producción.

### Resultados principales

| Hipótesis | Resultado |
|---|---|
| H1 — Modelo vs. referencia | XGBoost: WMAPE 12,0 % frente a 18,3 % del naïve estacional (mejora del 34 %) |
| H2 — Variables exógenas | El calendario especial reduce el error en días especiales (21,9 % → 18,4 %); la meteorología aporta poco una vez incluidos retardos y calendario |
| H3 — Modelo por sucursal | Mejora marginal (1–2 %) frente al modelo único con la sucursal como variable |
| H4 — Desperdicio | A igualdad de nivel de servicio, la merma simulada baja del 14,1 % al 6,4 % (−54,5 %) |

## Estructura del repositorio

```
├── data/
│   ├── kiel_dataset1/        Ventas diarias de una panadería de Kiel (2013–2019),
│   │                         meteorología y fechas de la Kieler Woche
│   ├── kiel_dataset2/        Facturación diaria de 3 sucursales (2015–2020), Meteolytix
│   └── procesado/            Conjuntos de datos con variables generadas (salida de scripts)
├── src/
│   ├── preparar_datos.py         Preparación e ingeniería de variables (CRISP-DM fase 3)
│   ├── experimento_ds1.py        Backtesting principal: baselines, SARIMA, Prophet, XGBoost, LightGBM
│   ├── ablacion_h2.py            Estudio de ablación de variables exógenas (H2)
│   ├── h3_sucursales.py          Modelo único vs. modelo por sucursal (H3)
│   └── simulacion_desperdicio.py Simulación de la política de producción (H4)
└── analisis/
    ├── resultados/           Métricas y predicciones (CSV)
    └── figuras/              Figuras de la memoria (PNG)
```

## Requisitos y ejecución

Python ≥ 3.11. Instalación de dependencias:

```bash
pip install -r requirements.txt
```

Ejecución (en orden):

```bash
python src/preparar_datos.py          # genera data/procesado/dataset1_features.csv
python src/experimento_ds1.py         # backtesting principal (~15 min en CPU)
python src/ablacion_h2.py             # requiere el paso anterior
python src/simulacion_desperdicio.py  # requiere experimento_ds1
python src/h3_sucursales.py           # independiente (usa dataset 2)
```

Los resultados se escriben en `analisis/resultados/` y las figuras en `analisis/figuras/`.

## Datos: procedencia y crédito

Los conjuntos de datos proceden del repositorio público de la iniciativa de educación abierta
[opencampus.sh](https://github.com/opencampus-sh/bakery-sales-project) (Kiel, Alemania), que los
publica con fines formativos; el segundo conjunto fue cedido por la empresa
[Meteolytix](https://meteolytix.de/) para un hackathon de dicha iniciativa. Se incluyen aquí
únicamente para garantizar la reproducibilidad del trabajo académico, con cita de la fuente.
Las cifras de ventas están anonimizadas/reescaladas por el proveedor original.

Los festivos de Schleswig-Holstein se obtienen con la biblioteca
[holidays](https://github.com/vacanza/holidays).

## Contexto académico

Este repositorio acompaña a la memoria del TFM y es requisito de reproducibilidad de la
titulación. El autor es el único responsable del código. El trabajo sigue la metodología
CRISP-DM y el protocolo de evaluación descrito en la memoria (backtesting con origen rodante,
WMAPE/MASE frente a baselines naïve).

## Licencia

Código bajo licencia MIT. Los datos conservan las condiciones de su fuente original
(ver sección de procedencia).
