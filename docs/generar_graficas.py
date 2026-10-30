"""
generar_graficas.py

Genera las graficas del README en version clara y oscura. GitHub elige
una u otra segun el tema de quien lo lea (etiqueta <picture>).

    python docs/generar_graficas.py

Los datos no estan copiados a mano: el conteo se simula con el
ContadorSentadillas real del proyecto, y la reconstruccion usa los
errores medidos que documenta ReconstructorPierna.
"""

import math
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "Proyecto-de-pose"))

from conteo_sentadillas import ANGULO_PROFUNDIDAD, ContadorSentadillas  # noqa: E402

SALIDA = os.path.join(AQUI, "img")

# Paleta de referencia de la skill de visualizacion, validada para
# daltonismo en los dos modos. Solo se usan los dos primeros puestos.
TEMAS = {
    "claro": dict(fondo="#ffffff", texto="#0b0b0b", secundario="#52514e",
                  rejilla="#e4e3df", serie1="#2a78d6", serie2="#eb6834"),
    "oscuro": dict(fondo="#0d1117", texto="#f0efec", secundario="#c3c2b7",
                   rejilla="#30302e", serie1="#3987e5", serie2="#d95926"),
}

# La misma tipografia de la app de escritorio. El texto se guarda como
# trazos, asi el SVG se ve igual en una maquina que no la tenga.
FUENTE = "Bahnschrift"
if not any(f.name == FUENTE for f in font_manager.fontManager.ttflist):
    FUENTE = "DejaVu Sans"

plt.rcParams.update({"font.family": FUENTE, "svg.fonttype": "path"})


# ---------------------------------------------------------------- datos

REPETICIONES = 5
FPS = 30
SEGUNDOS_POR_REPETICION = 2.5
ENSAYOS = 200


def serie_sentadillas(ruido, azar):
    """Angulo de rodilla de cinco sentadillas completas, con ruido."""
    angulos = []

    # Un segundo de pie antes y despues, como al grabar de verdad.
    for _ in range(FPS):
        angulos.append(170 + azar.gauss(0, ruido))

    pasos = int(FPS * SEGUNDOS_POR_REPETICION)

    for _ in range(REPETICIONES):
        for i in range(pasos):
            fase = i / pasos
            angulo = 127.5 + 42.5 * math.cos(2 * math.pi * fase)  # 170 a 85
            angulos.append(angulo + azar.gauss(0, ruido))

    for _ in range(FPS):
        angulos.append(170 + azar.gauss(0, ruido))

    return angulos


def contar_con_umbral(angulos, umbral=ANGULO_PROFUNDIDAD):
    """La version ingenua: una repeticion por cada cruce hacia abajo."""
    cuenta = 0

    for previo, actual in zip(angulos, angulos[1:]):
        if previo >= umbral > actual:
            cuenta += 1

    return cuenta


def contar_con_estados(angulos):
    contador = ContadorSentadillas()

    for angulo in angulos:
        contador.actualizar(angulo)

    return contador.repeticiones


def datos_conteo():
    azar = random.Random(7)
    ruidos = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    umbral, estados = [], []

    for ruido in ruidos:
        a = b = 0

        for _ in range(ENSAYOS):
            serie = serie_sentadillas(ruido, azar)
            a += contar_con_umbral(serie)
            b += contar_con_estados(serie)

        umbral.append(a / ENSAYOS)
        estados.append(b / ENSAYOS)

    return ruidos, umbral, estados


# Medido y documentado en perfil_corporal.ReconstructorPierna.
RECONSTRUCCION_SOLO_CAMARA = ([0, 0.5, 1, 2], [0.0, 30.2, 32.0, 38.8])
RECONSTRUCCION_CON_LASER = ([0, 1], [7.0, 18.0])


# ---------------------------------------------------------------- dibujo

def lienzo(t):
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=100)
    fig.patch.set_facecolor(t["fondo"])
    ax.set_facecolor(t["fondo"])

    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)

    ax.spines["bottom"].set_color(t["rejilla"])
    ax.tick_params(colors=t["secundario"], labelsize=10, length=0, pad=6)
    ax.grid(axis="y", color=t["rejilla"], linewidth=0.8)
    ax.set_axisbelow(True)

    return fig, ax


def titulos(fig, ax, t, titulo, subtitulo, eje_x, eje_y):
    fig.text(0.06, 0.95, titulo, color=t["texto"], fontsize=15,
             fontweight="bold", va="top")
    fig.text(0.06, 0.875, subtitulo, color=t["secundario"], fontsize=10.5,
             va="top")
    ax.set_xlabel(eje_x, color=t["secundario"], fontsize=10, labelpad=8)
    ax.set_ylabel(eje_y, color=t["secundario"], fontsize=10, labelpad=8)


def leyenda(ax, t):
    ley = ax.legend(loc="upper left", frameon=False, fontsize=10,
                    labelcolor=t["texto"], handlelength=1.6)
    return ley


def grafica_conteo(t, nombre, datos):
    ruidos, umbral, estados = datos
    fig, ax = lienzo(t)

    ax.axhline(REPETICIONES, color=t["secundario"], linewidth=1,
               linestyle=(0, (4, 3)))
    ax.text(10.15, REPETICIONES, "5 reales", color=t["secundario"],
            fontsize=9.5, va="center")

    ax.plot(ruidos, umbral, color=t["serie2"], linewidth=2,
            marker="o", markersize=6, markeredgecolor=t["fondo"],
            markeredgewidth=2, label="Umbral simple")
    ax.plot(ruidos, estados, color=t["serie1"], linewidth=2,
            marker="o", markersize=6, markeredgecolor=t["fondo"],
            markeredgewidth=2, label="Máquina de estados (la app)")

    ax.set_xlim(-0.3, 11.2)
    ax.set_ylim(0, max(umbral) * 1.12)
    ax.set_xticks(ruidos)
    leyenda(ax, t)

    titulos(fig, ax, t, "Contar sentadillas con ruido en el ángulo",
            "Repeticiones contadas en 5 sentadillas reales, promedio de "
            f"{ENSAYOS} ensayos simulados",
            "Ruido del ángulo de rodilla (desviación, grados)",
            "Repeticiones contadas")

    fig.subplots_adjust(left=0.1, right=0.9, top=0.78, bottom=0.15)
    fig.savefig(os.path.join(SALIDA, f"{nombre}.svg"),
                facecolor=t["fondo"])
    plt.close(fig)


def grafica_reconstruccion(t, nombre):
    fig, ax = lienzo(t)

    x, y = RECONSTRUCCION_SOLO_CAMARA
    ax.plot(x, y, color=t["serie1"], linewidth=2, marker="o", markersize=7,
            markeredgecolor=t["fondo"], markeredgewidth=2,
            label="Solo cámara")
    x2, y2 = RECONSTRUCCION_CON_LASER
    ax.plot(x2, y2, color=t["serie2"], linewidth=2, marker="s", markersize=7,
            markeredgecolor=t["fondo"], markeredgewidth=2,
            label="Cámara + láser en el tobillo")

    # Etiquetas solo donde cuentan la historia: el salto con medio pixel.
    ax.annotate("30.2°", (0.5, 30.2), textcoords="offset points",
                xytext=(0, 9), ha="center", color=t["texto"], fontsize=10)
    ax.annotate("0°", (0, 0), textcoords="offset points",
                xytext=(10, 4), ha="left", color=t["texto"], fontsize=10)

    ax.set_xlim(-0.1, 2.15)
    ax.set_ylim(0, 45)
    ax.set_xticks([0, 0.5, 1, 1.5, 2])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}°")
    leyenda(ax, t)

    titulos(fig, ax, t, "Reconstrucción 3D con una sola cámara",
            "Exacta con datos perfectos, inutilizable con medio píxel de "
            "ruido: por eso la cámara va de perfil",
            "Ruido en los puntos del cuerpo (píxeles)",
            "Error medio, ángulo de rodilla")

    fig.subplots_adjust(left=0.1, right=0.95, top=0.78, bottom=0.15)
    fig.savefig(os.path.join(SALIDA, f"{nombre}.svg"),
                facecolor=t["fondo"])
    plt.close(fig)


def main():
    os.makedirs(SALIDA, exist_ok=True)
    conteo = datos_conteo()

    print("ruido  umbral  estados")
    for fila in zip(*conteo):
        print("%5s  %6.2f  %7.2f" % fila)

    for modo, t in TEMAS.items():
        grafica_conteo(t, f"conteo_{modo}", conteo)
        grafica_reconstruccion(t, f"reconstruccion_{modo}")

    print("Graficas en", SALIDA)


if __name__ == "__main__":
    main()
