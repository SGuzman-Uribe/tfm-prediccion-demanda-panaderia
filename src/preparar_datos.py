# -*- coding: utf-8 -*-
"""Preparación de datos e ingeniería de variables (fase 3 de CRISP-DM).

Aquí construyo el conjunto de datos procesado del dataset 1, la panadería de Kiel.
La idea es juntar en una sola tabla todo lo que el panadero podría saber la noche
antes de hornear: las ventas pasadas, el tiempo que va a hacer, si hay Kieler Woche
y si el día es festivo en Schleswig-Holstein. A partir de ahí genero las variables
de calendario, los retardos y las medias móviles que después usan los modelos.
"""
import os
import pandas as pd
import numpy as np
import holidays

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATOS = os.path.join(RAIZ, "data")
SALIDA = os.path.join(DATOS, "procesado")
os.makedirs(SALIDA, exist_ok=True)

GRUPOS = {1: "Pan", 2: "Panecillos", 3: "Croissant",
          4: "Confitería", 5: "Tarta", 6: "Pan estacional"}


def cargar_crudos():
    # Ojo: el archivo de meteorología se llama "wheather.csv" (con la errata original del repositorio)
    d1 = pd.read_csv(os.path.join(DATOS, "kiel_dataset1", "sales.csv"),
                     encoding="utf-8-sig", parse_dates=["Date"])
    clima = pd.read_csv(os.path.join(DATOS, "kiel_dataset1", "wheather.csv"),
                        encoding="utf-8-sig", parse_dates=["Date"])
    kiwo = pd.read_csv(os.path.join(DATOS, "kiel_dataset1", "kiwo.csv"),
                       encoding="utf-8-sig", parse_dates=["Date"])
    return d1, clima, kiwo


def construir_features():
    ventas, clima, kiwo = cargar_crudos()

    # --- calendario base: todas las fechas del rango para cada grupo ---
    # Lo hago así para que los días sin venta registrada aparezcan como NaN y no
    # desaparezcan sin más; si no, los retardos se desplazarían y mezclarían días.
    fechas = pd.date_range(ventas.Date.min(), ventas.Date.max(), freq="D")
    base = pd.MultiIndex.from_product([fechas, sorted(GRUPOS)],
                                      names=["Date", "Group"]).to_frame(index=False)
    df = base.merge(ventas, on=["Date", "Group"], how="left")

    # --- variables de calendario ---
    df["dia_semana"] = df.Date.dt.dayofweek
    df["mes"] = df.Date.dt.month
    df["dia_mes"] = df.Date.dt.day
    df["semana_anyo"] = df.Date.dt.isocalendar().week.astype(int)

    # Festivos de Schleswig-Holstein, que es el estado federado de Kiel.
    # Además del festivo en sí, marco la víspera y el día siguiente: la gente
    # compra distinto el día antes de una fiesta y el día después.
    anyos = range(ventas.Date.min().year, ventas.Date.max().year + 1)
    fest_sh = holidays.Germany(subdiv="SH", years=anyos)
    fechas_fest = pd.to_datetime(sorted(fest_sh.keys()))
    df["festivo"] = df.Date.isin(fechas_fest).astype(int)
    df["vispera_festivo"] = df.Date.isin(fechas_fest - pd.Timedelta(days=1)).astype(int)
    df["post_festivo"] = df.Date.isin(fechas_fest + pd.Timedelta(days=1)).astype(int)

    # Kieler Woche: el gran festival de la ciudad, unos diez días en junio
    df["kiwo"] = df.Date.isin(kiwo.Date).astype(int)

    # --- meteorología ---
    # Hay algunos días sin registro; los relleno interpolando en el tiempo (el clima
    # de un día se parece al del anterior y al del siguiente) y remato los extremos
    # con ffill/bfill para que no quede ningún hueco.
    clima = clima.rename(columns={"Cloudiness": "nubosidad", "Temperature": "temperatura",
                                  "Wind Speed": "viento", "Weather Code": "codigo_tiempo"})
    clima = clima.set_index("Date").reindex(fechas)
    clima[["nubosidad", "temperatura", "viento"]] = (
        clima[["nubosidad", "temperatura", "viento"]].interpolate().ffill().bfill())
    # El código de tiempo sigue la tabla WMO: de 50 en adelante hay precipitación en
    # superficie (llovizna, lluvia, nieve, chubascos o tormenta). Lo convierto en un
    # indicador binario de "llueve o no", que es lo que de verdad importa para salir a comprar pan.
    clima["precipitacion"] = (clima["codigo_tiempo"] >= 50).fillna(False).astype(int)
    clima = clima.drop(columns=["codigo_tiempo"]).reset_index(names="Date")
    df = df.merge(clima, on="Date", how="left")

    # --- retardos y medias móviles por grupo ---
    # Todo va desplazado un día (shift(1)) a propósito: la madrugada en que se decide
    # la producción todavía no se conoce la venta de hoy. Sin ese desplazamiento
    # estaría colando información del futuro y los resultados serían demasiado bonitos.
    df = df.sort_values(["Group", "Date"]).reset_index(drop=True)
    g = df.groupby("Group")["Sales"]
    for k in (1, 7, 14):
        df[f"lag_{k}"] = g.shift(k)
    df["media_7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
    df["media_30"] = g.transform(lambda s: s.shift(1).rolling(30, min_periods=10).mean())
    df["std_7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=3).std())

    df["nombre_grupo"] = df.Group.map(GRUPOS)
    return df


if __name__ == "__main__":
    df = construir_features()
    ruta = os.path.join(SALIDA, "dataset1_features.csv")
    df.to_csv(ruta, index=False)
    con_venta = df.Sales.notna().sum()
    print(f"Listo. He guardado {ruta}: {len(df)} filas ({con_venta} con venta registrada), "
          f"del {df.Date.min().date()} al {df.Date.max().date()}")
    print("Variables generadas:", [c for c in df.columns if c not in ("Date", "Group", "Sales", "nombre_grupo")])
    print("Días festivos de SH dentro del rango:", int(df[df.Group == 1].festivo.sum()))
    print("Días de Kieler Woche dentro del rango:", int(df[df.Group == 1].kiwo.sum()))
