"""
generar_icono.py

Dibuja el icono del proyecto y lo exporta para la app de escritorio y
para Camcel, desde una sola fuente: asi las dos se ven como una misma
herramienta y no se pueden desincronizar.

El motivo es el que ya pinta la app sobre el video: una figura de
perfil en sentadilla, con el tronco en verde como la linea de tronco
del panel, y un arco en la rodilla, que es el angulo que se mide.

    python generar_icono.py
"""

import math
import os

from PIL import Image, ImageDraw

AQUI = os.path.dirname(os.path.abspath(__file__))
CAMCEL_RES = r"C:\dev\camcel\android\app\src\main\res"

# Misma paleta que COLOR en app_pose.py y Paleta en tema.dart.
FONDO = "#1B2623"
FILETE = "#35443F"
TEXTO = "#E6EBE6"
CORRECTO = "#1F7A50"

LADO = 1024

# Pose de perfil mirando a la derecha: rodilla adelante, cadera atras,
# tronco algo inclinado y brazos al frente para equilibrar.
CABEZA = (575, 255)
HOMBRO = (515, 360)
MANO = (760, 430)
CADERA = (385, 575)
RODILLA = (640, 655)
TOBILLO = (470, 815)
PUNTA_PIE = (625, 815)


def _arco_rodilla(dibujo, escala):
    """Arco del angulo de rodilla, entre el muslo y la tibia."""

    def angulo_hacia(p):
        return math.degrees(math.atan2(p[1] - RODILLA[1], p[0] - RODILLA[0]))

    a_muslo = angulo_hacia(CADERA)
    a_tibia = angulo_hacia(TOBILLO)

    radio = 168 * escala
    caja = [
        RODILLA[0] * escala - radio, RODILLA[1] * escala - radio,
        RODILLA[0] * escala + radio, RODILLA[1] * escala + radio,
    ]

    # PIL barre en sentido horario de start a end. Entre las dos
    # direcciones hay dos arcos posibles; el angulo de la rodilla es el
    # interior, el menor de 180. Con solo ordenarlos salia el exterior
    # y el arco parecia un circulo casi completo.
    inicio, fin = a_muslo, a_tibia

    if (fin - inicio) % 360 > 180:
        inicio, fin = fin, inicio

    if fin < inicio:
        fin += 360
    dibujo.arc(caja, start=inicio, end=fin, fill=CORRECTO,
               width=round(28 * escala))


def dibujar(lado=LADO):
    """Icono cuadrado de `lado` pixeles."""

    escala_dibujo = 4  # se dibuja grande y se reduce: bordes limpios
    px = lado * escala_dibujo
    e = px / LADO

    imagen = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    d = ImageDraw.Draw(imagen)

    margen = 24 * e
    d.rounded_rectangle(
        [margen, margen, px - margen, px - margen],
        radius=210 * e, fill=FONDO, outline=FILETE, width=round(10 * e),
    )

    def p(punto):
        return (punto[0] * e, punto[1] * e)

    grueso = round(46 * e)

    _arco_rodilla(d, e)

    # Piernas y brazo, en el color del esqueleto de la app.
    for a, b in ((TOBILLO, PUNTA_PIE), (RODILLA, TOBILLO),
                 (CADERA, RODILLA), (HOMBRO, MANO)):
        d.line([p(a), p(b)], fill=TEXTO, width=grueso)

    # El tronco va en verde: es la linea que la app usa para medirlo.
    d.line([p(CADERA), p(HOMBRO)], fill=CORRECTO, width=round(56 * e))

    # Articulaciones, con un aro del fondo para separarlas del trazo.
    for punto in (TOBILLO, RODILLA, CADERA, HOMBRO, MANO):
        r = 36 * e
        x, y = p(punto)
        d.ellipse([x - r, y - r, x + r, y + r], fill=TEXTO,
                  outline=FONDO, width=round(10 * e))

    r = 66 * e
    x, y = p(CABEZA)
    d.ellipse([x - r, y - r, x + r, y + r], fill=TEXTO)

    return imagen.resize((lado, lado), Image.LANCZOS)


def main():
    base = dibujar(LADO)

    base.save(os.path.join(AQUI, "icono.png"))
    base.save(
        os.path.join(AQUI, "icono.ico"),
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48),
               (64, 64), (128, 128), (256, 256)],
    )

    densidades = {
        "mipmap-mdpi": 48, "mipmap-hdpi": 72, "mipmap-xhdpi": 96,
        "mipmap-xxhdpi": 144, "mipmap-xxxhdpi": 192,
    }

    for carpeta, tam in densidades.items():
        destino = os.path.join(CAMCEL_RES, carpeta, "ic_launcher.png")

        if os.path.isdir(os.path.dirname(destino)):
            # Cada tamano se dibuja de nuevo en vez de reducir el grande:
            # a 48 px los trazos finos de un reescalado se emborronan.
            dibujar(tam).save(destino)

    print("Icono generado: icono.png, icono.ico y los mipmap de Camcel")


if __name__ == "__main__":
    main()
