# -*- coding: utf-8 -*-
"""Simulación de la política de producción (hipótesis H4).

La pregunta que quiero responder es muy de negocio: si el panadero horneara cada
día lo que dice el modelo más un colchón de seguridad, ¿cuánto pan sobraría y
cuánto faltaría? La merma simulada es lo que se produce de más respecto a la
demanda real, y la rotura, lo que se queda sin servir. Comparo la curva
merma-rotura de la política guiada por el mejor modelo con la de la política de
referencia (el naïve estacional, "hornear lo del mismo día de la semana pasada"),
sobre el mismo periodo de prueba.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(RAIZ, "analisis", "resultados")
FIG = os.path.join(RAIZ, "analisis", "figuras")

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

# Márgenes de seguridad que pruebo: del 0 % al 30 % en pasos de 2 puntos
MARGENES = np.arange(0.0, 0.31, 0.02)


def simular(d, margen):
    prod = d.yhat * (1 + margen)
    merma = np.maximum(0, prod - d.Sales)
    rotura = np.maximum(0, d.Sales - prod)
    return {
        "margen_%": 100 * margen,
        "merma_%": 100 * merma.sum() / prod.sum(),          # qué parte de lo horneado acaba sobrando
        "rotura_%": 100 * rotura.sum() / d.Sales.sum(),     # qué parte de la demanda se queda sin pan
    }


if __name__ == "__main__":
    preds = pd.read_csv(os.path.join(RES, "predicciones_ds1.csv"), parse_dates=["Date"])
    preds = preds.dropna(subset=["yhat"])
    # Dejo fuera el pan estacional: es producto de campaña y su producción sigue otra lógica
    # (se hornea por encargo o en fechas muy concretas), así que mezclarlo aquí no tendría sentido
    preds = preds[preds.Group != 6]

    mejor = (preds.groupby("modelo")
             .apply(lambda d: (d.Sales - d.yhat).abs().sum() / d.Sales.sum())
             .idxmin())
    print("El mejor modelo por WMAPE es:", mejor)

    curvas = []
    for modelo in [mejor, "snaive"]:
        d = preds[preds.modelo == modelo]
        for m in MARGENES:
            fila = simular(d, m)
            fila["politica"] = "modelo" if modelo == mejor else "referencia (naïve estacional)"
            curvas.append(fila)
    curvas = pd.DataFrame(curvas)
    curvas.to_csv(os.path.join(RES, "simulacion_desperdicio.csv"), index=False)

    # Comparación justa, a igualdad de servicio: fijo la rotura que tiene la política de
    # referencia y miro qué merma consigue el modelo para esa misma rotura, interpolando
    # sobre su curva. Si no igualara el servicio, podría "ganar" simplemente horneando menos.
    ref = curvas[curvas.politica.str.startswith("referencia")]
    mod = curvas[curvas.politica == "modelo"]
    ref0 = ref[ref["margen_%"] == 10.0].iloc[0]           # la referencia hornea con un colchón del 10 %
    orden = mod.sort_values("rotura_%")
    merma_mod_interp = np.interp(ref0["rotura_%"], orden["rotura_%"], orden["merma_%"])
    reduccion = 100 * (ref0["merma_%"] - merma_mod_interp) / ref0["merma_%"]
    resumen = pd.DataFrame([{
        "rotura_objetivo_%": ref0["rotura_%"],
        "merma_referencia_%": ref0["merma_%"],
        "merma_modelo_%": merma_mod_interp,
        "reduccion_merma_%": reduccion,
    }])
    resumen.to_csv(os.path.join(RES, "simulacion_resumen.csv"), index=False)
    print(resumen.round(2).to_string(index=False))

    # Figura: la frontera merma-rotura de las dos políticas. Cuanto más abajo y a la
    # izquierda esté una curva, mejor: menos pan tirado y menos clientes sin pan.
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for nombre, d in curvas.groupby("politica"):
        d = d.sort_values("rotura_%")
        ax.plot(d["rotura_%"], d["merma_%"], marker="o", ms=3, label=nombre)
    ax.set_xlabel("Rotura de stock (% de la demanda no servida)")
    ax.set_ylabel("Merma (% de la producción)")
    ax.set_title("Frontera merma-rotura según la política de producción (test, 5 categorías)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "f6_frontera_merma_rotura.png"))
    print("Figura guardada en f6_frontera_merma_rotura.png")
