# -*- coding: utf-8 -*-
"""Experimento principal sobre el dataset 1.

Backtesting de un paso adelante con reentrenamiento mensual sobre el último año
disponible (junio de 2018 a mayo de 2019).

El marco que intento imitar es el del propio negocio: cada madrugada el panadero
decide cuánto hornear sabiendo cómo se vendió hasta ayer, ni un día más. Por eso
toda la evaluación es a horizonte de 1 día, y por eso cada mes vuelvo a entrenar
con todo lo que ya se conocería en ese momento.

Modelos que comparo: los baselines (naïve, naïve estacional y media móvil de 4
semanas), SARIMA con variables exógenas, Prophet, y XGBoost y LightGBM entrenados
de forma conjunta para las seis categorías (multi-serie).
"""
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(RAIZ, "analisis", "resultados")
os.makedirs(RES, exist_ok=True)

GRUPOS = {1: "Pan", 2: "Panecillos", 3: "Croissant",
          4: "Confitería", 5: "Tarta", 6: "Pan estacional"}

INICIO_TEST = pd.Timestamp("2018-06-01")
FIN_TEST = pd.Timestamp("2019-05-31")

# Bloques de variables. Los separo así porque después, en el estudio de ablación,
# necesito poder quitar y poner cada bloque por separado.
FEATURES_CAL = ["dia_semana", "mes", "dia_mes", "semana_anyo",
                "festivo", "vispera_festivo", "post_festivo", "kiwo"]
FEATURES_CLIMA = ["nubosidad", "temperatura", "viento", "precipitacion"]
FEATURES_HIST = ["lag_1", "lag_7", "lag_14", "media_7", "media_30", "std_7"]
# A SARIMA le paso solo unas pocas exógenas: con demasiadas tarda muchísimo y se vuelve inestable
EXOG_SARIMA = ["festivo", "vispera_festivo", "kiwo", "temperatura", "precipitacion"]


def cargar():
    df = pd.read_csv(os.path.join(RAIZ, "data", "procesado", "dataset1_features.csv"),
                     parse_dates=["Date"])
    # Añado los retardos de 3 y 4 semanas, que no estaban en el procesado, para poder
    # construir la media móvil de 4 semanas: mi aproximación a "lo que haría el panadero a ojo"
    df = df.sort_values(["Group", "Date"]).reset_index(drop=True)
    g = df.groupby("Group")["Sales"]
    df["lag_21"] = g.shift(21)
    df["lag_28"] = g.shift(28)
    df["ma4_semanas"] = df[["lag_7", "lag_14", "lag_21", "lag_28"]].mean(axis=1)
    df["Group"] = df["Group"].astype(int)
    return df


def meses_de_test():
    return pd.date_range(INICIO_TEST, FIN_TEST, freq="MS")


def predecir_baselines(df):
    # Los baselines no se entrenan: son reglas fijas. Naïve = la venta de ayer,
    # naïve estacional = la del mismo día de la semana pasada, ma4 = media de las 4 últimas semanas.
    test = df[(df.Date >= INICIO_TEST) & (df.Date <= FIN_TEST) & df.Sales.notna()]
    out = []
    for nombre, col in [("naive", "lag_1"), ("snaive", "lag_7"), ("ma4", "ma4_semanas")]:
        p = test[["Date", "Group", "Sales", col]].rename(columns={col: "yhat"})
        p["modelo"] = nombre
        out.append(p)
    return pd.concat(out)


def predecir_sarima(df):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    out = []
    for grupo in sorted(GRUPOS):
        serie = (df[df.Group == grupo].set_index("Date")
                 .asfreq("D")[["Sales"] + EXOG_SARIMA])
        serie[EXOG_SARIMA] = serie[EXOG_SARIMA].ffill().fillna(0)
        for inicio_mes in meses_de_test():
            fin_mes = inicio_mes + pd.offsets.MonthEnd(0)
            train = serie[serie.index < inicio_mes]
            if train.Sales.notna().sum() < 120:
                # Con menos de cuatro meses de datos SARIMA no tiene de dónde aprender la estacionalidad
                continue
            try:
                mod = SARIMAX(train.Sales, exog=train[EXOG_SARIMA],
                              order=(1, 0, 1), seasonal_order=(1, 1, 1, 7),
                              enforce_stationarity=False, enforce_invertibility=False)
                res = mod.fit(disp=False, maxiter=200)
                # Truco importante: en lugar de predecir el mes entero de golpe, aplico el
                # modelo ya ajustado sobre la serie real hasta fin de mes y pido la predicción
                # a un paso (dynamic=False). Así cada día se predice conociendo la venta real
                # del día anterior, igual que haría el panadero.
                hasta_fin = serie[serie.index <= fin_mes]
                res2 = res.apply(hasta_fin.Sales, exog=hasta_fin[EXOG_SARIMA])
                pred = res2.get_prediction(start=inicio_mes, dynamic=False).predicted_mean
            except Exception as e:
                print(f"  SARIMA no ha podido ajustar el grupo {grupo} en el mes {inicio_mes.date()}: {e}")
                continue
            reales = serie.loc[inicio_mes:fin_mes, "Sales"]
            m = pd.DataFrame({"Date": reales.index, "Sales": reales.values,
                              "yhat": pred.reindex(reales.index).values})
            m["Group"] = grupo
            out.append(m.dropna(subset=["Sales"]))
        print(f"SARIMA terminado para el grupo {grupo} ({GRUPOS[grupo]})")
    r = pd.concat(out)
    r["modelo"] = "sarima"
    return r


def predecir_prophet(df):
    from prophet import Prophet
    import logging
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    # Prophet tiene su propio mecanismo para festivos, así que le paso los festivos de SH
    # y los días de Kieler Woche como dos "festivos" distintos en lugar de como regresores
    fechas_fest = df.loc[(df.Group == 1) & (df.festivo == 1), "Date"]
    fechas_kiwo = df.loc[(df.Group == 1) & (df.kiwo == 1), "Date"]
    festivos = pd.concat([
        pd.DataFrame({"holiday": "festivo_sh", "ds": fechas_fest}),
        pd.DataFrame({"holiday": "kieler_woche", "ds": fechas_kiwo}),
    ])
    out = []
    for grupo in sorted(GRUPOS):
        datos = df[(df.Group == grupo) & df.Sales.notna()][
            ["Date", "Sales", "temperatura", "precipitacion"]].rename(
            columns={"Date": "ds", "Sales": "y"})
        for inicio_mes in meses_de_test():
            fin_mes = inicio_mes + pd.offsets.MonthEnd(0)
            train = datos[datos.ds < inicio_mes]
            futuro = datos[(datos.ds >= inicio_mes) & (datos.ds <= fin_mes)]
            if len(train) < 120 or futuro.empty:
                continue
            m = Prophet(holidays=festivos, weekly_seasonality=True,
                        yearly_seasonality=True, daily_seasonality=False)
            m.add_regressor("temperatura")
            m.add_regressor("precipitacion")
            m.fit(train)
            pred = m.predict(futuro[["ds", "temperatura", "precipitacion"]])
            r = futuro[["ds", "y"]].copy()
            r["yhat"] = pred["yhat"].values
            r["Group"] = grupo
            out.append(r)
        print(f"Prophet terminado para el grupo {grupo} ({GRUPOS[grupo]})")
    r = pd.concat(out).rename(columns={"ds": "Date", "y": "Sales"})
    r["modelo"] = "prophet"
    return r


def predecir_boosting(df, cual="lightgbm", features=None, etiqueta=None):
    import lightgbm as lgb
    import xgboost as xgb
    # Por defecto uso todas las variables y meto "Group" como variable más: un único
    # modelo aprende las seis categorías a la vez y comparte lo que tienen en común
    features = features or (FEATURES_CAL + FEATURES_CLIMA + FEATURES_HIST + ["Group"])
    etiqueta = etiqueta or cual
    out = []
    for inicio_mes in meses_de_test():
        fin_mes = inicio_mes + pd.offsets.MonthEnd(0)
        # Quito las filas sin lag_28 para que todos los modelos entrenen exactamente con los mismos días
        train = df[(df.Date < inicio_mes) & df.Sales.notna()].dropna(subset=["lag_28"])
        test = df[(df.Date >= inicio_mes) & (df.Date <= fin_mes) & df.Sales.notna()]
        X_tr, y_tr = train[features], train.Sales
        X_te = test[features]
        if cual == "lightgbm":
            mod = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05,
                                    num_leaves=31, min_child_samples=20,
                                    subsample=0.9, colsample_bytree=0.9,
                                    random_state=42, verbose=-1)
            mod.fit(X_tr, y_tr, categorical_feature=["Group"] if "Group" in features else "auto")
        else:
            mod = xgb.XGBRegressor(n_estimators=600, learning_rate=0.05,
                                   max_depth=6, subsample=0.9, colsample_bytree=0.9,
                                   random_state=42, verbosity=0)
            mod.fit(X_tr, y_tr)
        r = test[["Date", "Group", "Sales"]].copy()
        # Una venta negativa no tiene sentido, así que recorto las predicciones en cero
        r["yhat"] = np.clip(mod.predict(X_te), 0, None)
        out.append(r)
    r = pd.concat(out)
    r["modelo"] = etiqueta
    print(f"{etiqueta} terminado")
    return r


def metricas(preds, df):
    """WMAPE, MASE y MAPE por modelo y grupo sobre el periodo de prueba.

    El WMAPE es mi métrica principal porque pondera por volumen y no se dispara con
    ventas pequeñas; el MAPE lo calculo igualmente por comparabilidad con la literatura.
    """
    # La escala del MASE es el MAE del naïve estacional (m = 7) en el periodo de entrenamiento:
    # un MASE < 1 significa "mejor que repetir lo de la semana pasada"
    train = df[(df.Date < INICIO_TEST) & df.Sales.notna()].dropna(subset=["lag_7"])
    escala = train.assign(err=(train.Sales - train.lag_7).abs()).groupby("Group").err.mean()
    filas = []
    for (modelo, grupo), d in preds.groupby(["modelo", "Group"]):
        d = d.dropna(subset=["yhat"])
        err = (d.Sales - d.yhat).abs()
        filas.append({
            "modelo": modelo, "grupo": GRUPOS[grupo],
            "WMAPE_%": 100 * err.sum() / d.Sales.sum(),
            "MASE": err.mean() / escala[grupo],
            "MAPE_%": 100 * (err / d.Sales).mean(),
            "n_dias": len(d),
        })
    por_grupo = pd.DataFrame(filas)
    # Agregado global ponderado por ventas, con y sin el pan estacional. Lo calculo de las dos
    # formas porque el pan estacional es intermitente y puede distorsionar la comparación.
    filas_g = []
    for modelo, d in preds.dropna(subset=["yhat"]).groupby("modelo"):
        for ambito, dd in [("global", d), ("sin_estacional", d[d.Group != 6])]:
            err = (dd.Sales - dd.yhat).abs()
            filas_g.append({"modelo": modelo, "ambito": ambito,
                            "WMAPE_%": 100 * err.sum() / dd.Sales.sum()})
    agregado = pd.DataFrame(filas_g)
    return por_grupo, agregado


if __name__ == "__main__":
    df = cargar()
    preds = pd.concat([
        predecir_baselines(df),
        predecir_sarima(df),
        predecir_prophet(df),
        predecir_boosting(df, "xgboost"),
        predecir_boosting(df, "lightgbm"),
    ])
    preds.to_csv(os.path.join(RES, "predicciones_ds1.csv"), index=False)
    por_grupo, agregado = metricas(preds, df)
    por_grupo.to_csv(os.path.join(RES, "metricas_ds1_por_grupo.csv"), index=False)
    agregado.to_csv(os.path.join(RES, "metricas_ds1_agregado.csv"), index=False)
    print("\n=== WMAPE global por modelo (con y sin pan estacional) ===")
    print(agregado.pivot(index="modelo", columns="ambito", values="WMAPE_%").round(2))
    print("\n=== WMAPE por grupo y modelo ===")
    print(por_grupo.pivot(index="grupo", columns="modelo", values="WMAPE_%").round(1))
