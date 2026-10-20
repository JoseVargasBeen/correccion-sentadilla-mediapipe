"""
calibrar_encorvamiento.py

Mide el indice de encorvamiento real con TU camara y TU cuerpo, y
propone un umbral para UMBRAL_JOROBA en app_pose.py.

Por que asi y no con fotos de internet: el indice depende del angulo
de camara, de la distancia y de las proporciones de cada persona. Un
umbral sacado de fotos ajenas, tomadas con otra lente y otra altura de
camara, no vale para este montaje. Dos minutos delante de la camara
dan un dato mejor que cualquier galeria.

Como se usa:

    python calibrar_encorvamiento.py               # webcam 0
    python calibrar_encorvamiento.py --fuente 1
    python calibrar_encorvamiento.py --fuente http://192.168.1.8:8080/video

Colocate de PERFIL a la camara, de cuerpo entero o al menos de la
cadera hacia arriba, y sigue las instrucciones en pantalla: primero
derecho, luego encorvado. Pulsa q para abortar.
"""

import argparse
import statistics
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from app_pose import (
    POSE_MODEL_PATH,
    VISIBILIDAD_MINIMA,
    LectorCamara,
    lado_visible,
    medidas_de,
)

SEGUNDOS_POR_FASE = 12
SEGUNDOS_PREPARACION = 5

FASES = [
    ("derecho", "Ponte DERECHO, oreja sobre el hombro"),
    ("encorvado", "ENCORVATE como cuando te cansas"),
]


def medir_indice(points, landmarks):
    """Indice de encorvamiento de un frame, o None si no se ve bien.

    Usa medidas_de de app_pose, la misma que usa la app: si midieran
    distinto, el umbral que sale de aqui no valdria alli.
    """
    lado, visibilidad = lado_visible(landmarks)

    if visibilidad < VISIBILIDAD_MINIMA:
        return None

    return medidas_de(points, lado)["joroba"]


def puntos_de(result, ancho, alto):
    if not result.pose_landmarks:
        return None

    return [
        (int(lm.x * ancho), int(lm.y * alto))
        for lm in result.pose_landmarks[0]
    ]


def rotulo(imagen, lineas):
    for i, (texto, color) in enumerate(lineas):
        cv2.putText(imagen, texto, (20, 40 + i * 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(imagen, texto, (20, 40 + i * 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def recoger(cap, detector, inicio, etiqueta, instruccion):
    """Graba una fase y devuelve la lista de indices medidos."""
    muestras = []
    fase_inicio = None

    while True:
        ok, frame = cap.leer()

        if not cap.corriendo:
            raise RuntimeError("Se corto la senal de la camara")

        if not ok:
            time.sleep(0.002)
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(
            imagen, int((time.time() - inicio) * 1000)
        )

        alto, ancho = frame.shape[:2]
        points = puntos_de(result, ancho, alto)
        indice = (
            medir_indice(points, result.pose_landmarks[0]) if points else None
        )

        ahora = time.time()

        if fase_inicio is None:
            # No se empieza a contar hasta ver a la persona entera: si
            # no, los primeros segundos entran como muestras vacias.
            if indice is None:
                rotulo(frame, [("Colocate de perfil ante la camara",
                                (0, 200, 255))])
                cv2.imshow("Calibracion", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt
                continue
            fase_inicio = ahora

        transcurrido = ahora - fase_inicio
        restante = SEGUNDOS_PREPARACION + SEGUNDOS_POR_FASE - transcurrido

        if restante <= 0:
            break

        grabando = transcurrido >= SEGUNDOS_PREPARACION

        if grabando and indice is not None:
            muestras.append(indice)

        if points:
            for p in points:
                cv2.circle(frame, p, 3, (255, 255, 255), -1)

        estado = "GRABANDO" if grabando else "preparate"
        color = (80, 220, 120) if grabando else (0, 200, 255)
        rotulo(frame, [
            (instruccion, (255, 255, 255)),
            (f"{estado}  {restante:4.1f}s  muestras: {len(muestras)}", color),
            (f"indice: {indice:.3f}" if indice is not None
             else "no te veo entero", (255, 255, 255)),
        ])

        cv2.imshow("Calibracion", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt

    return muestras


def resumen(nombre, muestras):
    media = statistics.mean(muestras)
    desv = statistics.pstdev(muestras)
    print(f"  {nombre:10} n={len(muestras):4d}  "
          f"media={media:+.3f}  sd={desv:.3f}  "
          f"min={min(muestras):+.3f}  max={max(muestras):+.3f}")
    return media, desv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fuente", default="0",
                        help="indice de webcam o URL de video")
    args = parser.parse_args()

    fuente = int(args.fuente) if args.fuente.isdigit() else args.fuente

    cap = LectorCamara(fuente)
    cap.iniciar()

    # LectorCamara abre la fuente dentro de su propio hilo, asi que
    # conectar no es inmediato: hay que esperar a que confirme.
    print(f"Conectando con {fuente} ...")

    while not cap.conectado and cap.corriendo:
        time.sleep(0.05)

    if not cap.conectado:
        print(cap.error or f"No se pudo abrir la camara: {fuente}")
        return 1

    opciones = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    detector = vision.PoseLandmarker.create_from_options(opciones)
    inicio = time.time()

    recogido = {}

    try:
        for etiqueta, instruccion in FASES:
            print(f"\n=== {instruccion} ===")
            muestras = recoger(cap, detector, inicio, etiqueta, instruccion)

            if len(muestras) < 20:
                print(f"Solo {len(muestras)} muestras utiles en la fase "
                      f"'{etiqueta}'. Repite con mejor encuadre.")
                return 1

            recogido[etiqueta] = muestras
    except KeyboardInterrupt:
        print("\nCalibracion cancelada.")
        return 1
    except RuntimeError as error:
        print(f"\n{error}")
        return 1
    finally:
        cap.detener()
        detector.close()
        cv2.destroyAllWindows()

    print("\n--- Medido ---")
    media_d, desv_d = resumen("derecho", recogido["derecho"])
    media_e, desv_e = resumen("encorvado", recogido["encorvado"])

    # Se trabaja en valor absoluto porque app_pose.py compara con abs().
    # medidas_de ya orienta el signo hacia adelante, pero una cabeza
    # echada hacia atras tambien es una postura a avisar.
    abs_d = abs(media_d)
    abs_e = abs(media_e)

    print("\n--- Resultado ---")

    if abs_e <= abs_d + 0.01:
        print("Las dos posturas dan casi lo mismo. La camara no esta "
              "captando la diferencia: ponte mas de perfil, o mas cerca, "
              "y repite.")
        return 1

    # Punto medio ponderado por la dispersion de cada grupo: el umbral
    # cae mas cerca del grupo que menos varia.
    peso_d = desv_d + 1e-6
    peso_e = desv_e + 1e-6
    umbral = (abs_d * peso_e + abs_e * peso_d) / (peso_d + peso_e)

    separacion = (abs_e - abs_d) / (desv_d + desv_e + 1e-6)

    print(f"  derecho   |indice| = {abs_d:.3f}")
    print(f"  encorvado |indice| = {abs_e:.3f}")
    print(f"  separacion         = {separacion:.1f} desviaciones")
    print()
    print(f"  UMBRAL_JOROBA = {umbral:.2f}")

    if separacion < 2:
        print("\n  Ojo: las dos nubes se solapan bastante. El umbral "
              "va a dar falsos avisos. Prueba a repetir mas de perfil.")

    print("\nPon ese valor en app_pose.py, en UMBRAL_JOROBA.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
