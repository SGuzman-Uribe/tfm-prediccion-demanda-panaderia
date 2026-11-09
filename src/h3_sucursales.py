# -*- coding: utf-8 -*-
"""Hipótesis H3 (dataset 2): ¿conviene un modelo por sucursal o uno único?

Comparo un LightGBM entrenado de forma conjunta para las tres sucursales (con la
sucursal como variable más) frente a tres modelos independientes, uno por tienda.
La intuición detrás de H3 es que las tiendas no se parecen tanto como uno pensaría:
la de la playa vende muchísimo más en verano y las otras dos son casi planas.

La serie se corta el 29/02/2020 para dejar fuera la pandemia, que cambia las
reglas del juego y no es el fenómeno que quiero estudiar. La evaluación es un
backtesting de un paso adelante con reentrenamiento mensual sobre el último año
(marzo de 2019 a febrero de 2020).
"""
import os
import warnings
import numpy as np
import pandas as pd
import holidays
import lightgbm as lgb

warnings.filterwarnings("ignore")

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(RAIZ, "analisis", "resultados")
os.makedirs(RES, exist_ok=True)

CORTE = pd.Timestamp("2020-02-29")
INICIO_TEST = pd.Timestamp("2019-03-01")

# Mismas variables que en el dataset 1, salvo la meteorología (este dataset no la trae)
# y con el grupo de artículo en lugar de la categoría
FEATS = ["dia_semana", "mes", "dia_mes", "semana_anyo",
         "festivo", "vispera_festivo", "post_festivo", "kiwo",
         "lag_1", "lag_7", "lag_14", "media_7", "media_30", "std_7",
         "Artikelgruppe"]


def preparar():
    # El dataset 2 viene con las columnas en alemán (Datum = fecha, Umsatz = facturación,
    # Filiale = sucursal, Artikelgruppe = grupo de artículo); las renombro a las del dataset 1
    d2 = pd.read_csv(os.path.join(RAIZ, "data", "kiel_dataset2", "ventas_multitienda.csv"),
                     encoding="utf-8-sig", parse_dates=["Datum"])
    kiwo = pd.read_csv(os.path.join(RAIZ, "data", "kiel_dataset1", "kiwo.csv"),
                       encoding="utf-8-sig", parse_dates=["Date"])
    d2 = d2[d2.Datum <= CORTE].rename(columns={"Datum": "Date", "Umsatz": "Sales"})

    # Rejilla completa fecha × sucursal × grupo, por el mismo motivo que en el dataset 1:
    # que los días sin registro queden como NaN y no descoloquen los retardos
    fechas = pd.date_range(d2.Date.min(), d2.Date.max(), freq="D")
    base = pd.MultiIndex.from_product(
        [fechas, sorted(d2.Filiale.unique()), sorted(d2.Artikelgruppe.unique())],
        names=["Date", "Filiale", "Artikelgruppe"]).to_frame(index=False)
    df = base.merge(d2, on=["Date", "Filiale", "Artikelgruppe"], how="left")

    df["dia_semana"] = df.Date.dt.dayofweek
    df["mes"] = df.Date.dt.month
    df["dia_mes"] = df.Date.dt.day
    df["semana_anyo"] = df.Date.dt.isocalendar().week.astype(int)
    fest = holidays.Germany(subdiv="SH", years=range(2015, 2021))
    ff = pd.to_datetime(sorted(fest.keys()))
    df["festivo"] = df.Date.isin(ff).astype(int)
    df["vispera_festivo"] = df.Date.isin(ff - pd.Timedelta(days=1)).astype(int)
    df["post_festivo"] = df.Date.isin(ff + pd.Timedelta(days=1)).astype(int)
    df["kiwo"] = df.Date.isin(kiwo.Date).astype(int)

    # Retardos y medias móviles, siempre desplazados un día para no usar información del futuro.
    # Aquí el groupby va por sucursal Y grupo: la historia de una tienda no debe contaminar la de otra.
    df = df.sort_values(["Filiale", "Artikelgruppe", "Date"]).reset_index(drop=True)
    g = df.groupby(["Filiale", "Artikelgruppe"])["Sales"]
    for k in (1, 7, 14):
        df[f"lag_{k}"] = g.shift(k)
    df["media_7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
    df["media_30"] = g.transform(lambda s: s.shift(1).rolling(30, min_periods=10).mean())
    df["std_7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=3).std())
    return df


def backtest(df, pooled=True):
    out = []
    for inicio_mes in pd.date_range(INICIO_TEST, CORTE, freq="MS"):
        fin_mes = min(inicio_mes + pd.offsets.MonthEnd(0), CORTE)
        test = df[(df.Date >= inicio_mes) & (df.Date <= fin_mes) & df.Sales.notna()]
        if pooled:
            # Modelo único: entrena con las tres tiendas a la vez y recibe la sucursal como categórica
            train = df[(df.Date < inicio_mes) & df.Sales.notna()].dropna(subset=["lag_14"])
            mod = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31,
                                    subsample=0.9, colsample_bytree=0.9,
                                    random_state=42, verbose=-1)
            mod.fit(train[FEATS + ["Filiale"]], train.Sales,
                    categorical_feature=["Artikelgruppe", "Filiale"])
            r = test[["Date", "Filiale", "Artikelgruppe", "Sales"]].copy()
            r["yhat"] = np.clip(mod.predict(test[FEATS + ["Filiale"]]), 0, None)
            out.append(r)
        else:
            # Un modelo por sucursal: cada tienda solo ve sus propios datos. Tiene menos
            # historia para aprender, pero no tiene que "promediar" comportamientos distintos.
            for fil in sorted(df.Filiale.unique()):
                train = df[(df.Filiale == fil) & (df.Date < inicio_mes)
                           & df.Sales.notna()].dropna(subset=["lag_14"])
                te = test[test.Filiale == fil]
                mod = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31,
                                        subsample=0.9, colsample_bytree=0.9,
                                        random_state=42, verbose=-1)
                mod.fit(train[FEATS], train.Sales, categorical_feature=["Artikelgruppe"])
                r = te[["Date", "Filiale", "Artikelgruppe", "Sales"]].copy()
                r["yhat"] = np.clip(mod.predict(te[FEATS]), 0, None)
                out.append(r)
    r = pd.concat(out)
    r["estrategia"] = "modelo_unico" if pooled else "modelo_por_sucursal"
    return r


if __name__ == "__main__":
    df = preparar()
    print("Filas preparadas:", len(df), "| periodo de test del", INICIO_TEST.date(), "al", CORTE.date())
    preds = pd.concat([backtest(df, pooled=True), backtest(df, pooled=False)])
    preds.to_csv(os.path.join(RES, "predicciones_h3.csv"), index=False)
    filas = []
    for (estrategia, fil), d in preds.groupby(["estrategia", "Filiale"]):
        err = (d.Sales - d.yhat).abs()
        filas.append({"estrategia": estrategia, "sucursal": fil,
                      "WMAPE_%": 100 * err.sum() / d.Sales.sum()})
    tabla = pd.DataFrame(filas).pivot(index="sucursal", columns="estrategia", values="WMAPE_%")
    # Positivo = el modelo por sucursal gana; negativo = el modelo único lo hace mejor
    tabla["mejora_por_sucursal_%"] = 100 * (tabla.modelo_unico - tabla.modelo_por_sucursal) / tabla.modelo_unico
    tabla.round(2).to_csv(os.path.join(RES, "h3_resultados.csv"))
    print(tabla.round(2).to_string())
