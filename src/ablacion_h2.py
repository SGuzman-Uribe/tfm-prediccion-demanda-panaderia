# -*- coding: utf-8 -*-
"""Estudio de ablación (hipótesis H2): ¿cuánto aporta cada bloque de variables exógenas?

La idea es ir añadiendo bloques al LightGBM, uno cada vez, y ver cuánto baja el error.
Me interesa sobre todo lo que pasa en los días especiales (festivos, vísperas y Kieler
Woche), porque son los días en que equivocarse sale más caro y donde, según la
literatura, el calendario y la meteorología deberían marcar la diferencia.

Configuraciones: (a) solo historia y calendario básico; (b) lo anterior más el
calendario especial (festivos, vísperas, Kieler Woche); (c) todo lo anterior más la
meteorología, es decir, el modelo completo.
"""
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Reutilizo la carga de datos y el backtesting de boosting del experimento principal
# para que las comparaciones sean exactamente con el mismo procedimiento
from experimento_ds1 import (cargar, predecir_boosting, GRUPOS,
                             INICIO_TEST, FIN_TEST,
                             FEATURES_CAL, FEATURES_CLIMA, FEATURES_HIST)

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(RAIZ, "analisis", "resultados")

CAL_BASICO = ["dia_semana", "mes", "dia_mes", "semana_anyo"]
CAL_ESPECIAL = ["festivo", "vispera_festivo", "post_festivo", "kiwo"]

CONFIGS = {
    "a_historia": FEATURES_HIST + CAL_BASICO + ["Group"],
    "b_mas_calendario": FEATURES_HIST + CAL_BASICO + CAL_ESPECIAL + ["Group"],
    "c_completo": FEATURES_HIST + CAL_BASICO + CAL_ESPECIAL + FEATURES_CLIMA + ["Group"],
}


def wmape(d):
    return 100 * (d.Sales - d.yhat).abs().sum() / d.Sales.sum()


if __name__ == "__main__":
    df = cargar()
    # Marco como "especial" cualquier día que sea festivo, víspera de festivo o de Kieler Woche
    especiales = df[["Date", "festivo", "vispera_festivo", "kiwo"]].drop_duplicates("Date")
    especiales["especial"] = ((especiales.festivo + especiales.vispera_festivo
                               + especiales.kiwo) > 0).astype(int)

    filas, preds_todas = [], []
    for nombre, feats in CONFIGS.items():
        p = predecir_boosting(df, "lightgbm", features=feats, etiqueta=nombre)
        p = p.merge(especiales[["Date", "especial"]], on="Date")
        preds_todas.append(p)
        # El mismo error partido en tres: global, solo días especiales y solo días normales
        filas.append({
            "configuracion": nombre,
            "WMAPE_global_%": wmape(p),
            "WMAPE_dias_especiales_%": wmape(p[p.especial == 1]),
            "WMAPE_dias_normales_%": wmape(p[p.especial == 0]),
            "n_dias_especiales": p[p.especial == 1].Date.nunique(),
        })
    tabla = pd.DataFrame(filas)
    tabla.to_csv(os.path.join(RES, "ablacion_h2.csv"), index=False)
    pd.concat(preds_todas).to_csv(os.path.join(RES, "predicciones_ablacion.csv"), index=False)
    print(tabla.round(2).to_string(index=False))
