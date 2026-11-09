# Análisis exploratorio — datasets de panadería de Kiel

Generado: 2026-10-02

## Dataset 1 — panadería de Kiel (sales.csv)
- Filas: 10869 | Periodo: 2013-07-01 a 2019-05-31 | Días únicos: 2115
- Días sin ningún registro: 46 de 2161

Ventas diarias por grupo (media / mediana / desv. típ. / nº días):

                 mean  median    std  count
Pan             124.3   121.7   40.9   2115
Panecillos      398.8   367.8  137.0   2115
Croissant       164.3   143.7   75.3   2115
Confitería       87.6    80.0   34.3   2061
Tarta           278.2   268.2   99.1   2115
Pan estacional   67.2    63.2   31.9    348

- Estacionalidad semanal fuerte: vender el domingo supone de media un 20 % más que el lunes (promedio entre grupos).

## Efecto Kieler Woche (junio, días KiWo vs resto de junio)

KielerWoche    0.0    1.0  efecto_%
Pan          129.3  136.0       5.2
Panecillos   412.9  538.6      30.4
Croissant    165.0  223.3      35.3
Confitería    71.2   81.2      13.9
Tarta        278.1  315.7      13.5

- Efecto medio de la Kieler Woche sobre las ventas de junio: +20 % (varía por grupo).

## Clima
- Correlación ventas totales ~ temperatura: r = 0.47
- Correlación ventas totales ~ nubosidad: r = -0.19
- (La relación es por grupo y no lineal: motiva modelos con features, no solo correlaciones.)

## Dataset 2 — 3 sucursales (Meteolytix)
- Filas: 62190 | Periodo: 2015-01-02 a 2020-05-27

Ratio venta media verano/invierno por sucursal (clave para H3):
- Filiale 1 (playa): 0.99
- Filiale 2 (centro): 1.27
- Filiale 3 (periferia): 0.96

- COVID: el periodo mar–may 2020 muestra caída visible; decidir con el director si se excluye (recomendado: cortar en 2020-02-29).