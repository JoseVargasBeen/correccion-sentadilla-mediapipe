"""
pose_angulos_sentadilla.py

Extiende pose_tasks_webcam.py para calcular:
- Angulo de rodilla (profundidad de sentadilla).
- Angulo de cadera (inclinacion del tronco hacia adelante).
- Alineacion de la espalda respecto a la vertical (postura recta /
  inclinada) y una estimacion de encorvamiento (joroba) comparando
  la oreja contra la linea hombro-cadera.

Nota sobre la espalda:
MediaPipe Pose (lite/full/heavy) siempre entrega los mismos 33
landmarks; ningun tamano de modelo agrega puntos intermedios en la
columna. Por eso la postura de espalda se infiere con geometria a
partir de los puntos que ya existen (hombros, caderas, oreja), en
vez de leer puntos "de la columna" que el modelo no tiene.

Controles:
    q -> salir
"""

import math
import time

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =================================
# Configuracion
# =================================

# Fuente de video:
# - Webcam del PC: usar el indice de camara, ej. 0
# - Celular con IP Webcam (Android, por WiFi): usar la URL del stream,
#   ej. "http://192.168.1.5:8080/video"
#CAM_INDEX = 0
CAM_INDEX = "http://192.168.1.8:8080/video"

# IP Webcam manda el video con la orientacion fija de cuando arranco
# la app (no gira solo si rotas el celular). Si la imagen sale de
# lado o al reves, cambiar aqui:
#   None                            -> no rotar
#   cv2.ROTATE_90_CLOCKWISE         -> celular en vertical (mas comun)
#   cv2.ROTATE_90_COUNTERCLOCKWISE  -> celular en vertical, al otro lado
#   cv2.ROTATE_180                  -> imagen al reves
ROTAR_FRAME = cv2.ROTATE_90_CLOCKWISE

# Alto maximo (px) de la ventana mostrada. Si el frame (ej. tras
# rotar 90 grados) queda mas alto que esto, se escala para que
# quepa completo en pantalla. No afecta la deteccion, solo la
# visualizacion.
ALTO_MAX_VENTANA = 900

POSE_MODEL_PATH = "models/pose_landmarker_lite.task"

USE_SEGMENTATION = True
MASK_THRESH = 0.2


# =================================
# Indices de landmarks (MediaPipe Pose, 33 puntos)
# Lado derecho del sujeto (visible de frente a la camara)
# =================================

NARIZ = 0
OREJA_D = 8

HOMBRO_I = 11
HOMBRO_D = 12

CADERA_I = 23
CADERA_D = 24

RODILLA_D = 26
TOBILLO_D = 28


# =================================
# Umbrales (placeholder, pendientes de calibrar con datos reales)
# =================================

# Angulo respecto a la vertical (grados) a partir del cual se
# considera que el tronco esta "inclinado".
UMBRAL_INCLINACION = 20.0

# Desplazamiento horizontal oreja-vs-linea de hombros, normalizado
# por la longitud del tronco (hombro-cadera). Mayor valor = mas
# encorvado / cabeza adelantada.
UMBRAL_JOROBA = 0.20


# =================================
# Conexiones del skeleton (igual que pose_tasks_webcam.py)
# =================================

POSE_CONNECTIONS = [
    (0, 1), (0, 4),
    (1, 2), (2, 3), (3, 7),
    (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21),
    (17, 19),
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (27, 31),
    (28, 32),
    (29, 31),
    (30, 32),
]


# =================================
# Geometria
# =================================

def calcular_angulo(a, b, c):
    """Angulo en grados (0-180) en el vertice b, formado por los
    segmentos b->a y b->c."""

    ang_ba = math.atan2(a[1] - b[1], a[0] - b[0])
    ang_bc = math.atan2(c[1] - b[1], c[0] - b[0])

    angulo = math.degrees(ang_ba - ang_bc)
    angulo = abs(angulo)

    if angulo > 180:
        angulo = 360 - angulo

    return angulo


def punto_medio(p1, p2):
    return (
        (p1[0] + p2[0]) / 2,
        (p1[1] + p2[1]) / 2,
    )


def angulo_respecto_vertical(punto_inferior, punto_superior):
    """Angulo (grados) entre el segmento punto_inferior->punto_superior
    y la vertical. 0 = perfectamente vertical (espalda recta).
    Signo indica hacia que lado se inclina."""

    dx = punto_superior[0] - punto_inferior[0]
    dy = punto_superior[1] - punto_inferior[1]

    # En imagen, y crece hacia abajo -> "arriba" es -y.
    angulo = math.degrees(math.atan2(dx, -dy))

    return angulo


def indice_encorvamiento(oreja, hombro_medio, cadera_medio):
    """Distancia perpendicular de la oreja respecto a la linea
    cadera-hombro (extendida), normalizada por la longitud del
    tronco al cuadrado para que el resultado sea adimensional.

    Al medir perpendicular al TRONCO (no al horizontal de la
    imagen), el indice queda invariante a que el cuerpo entero se
    incline (ej. agacharse a recoger algo con la espalda recta):
    si tronco y cabeza rotan juntos, la oreja sigue sobre la linea
    y el indice se mantiene bajo. Solo sube cuando la cabeza/cuello
    se adelanta RESPECTO al tronco (encorvamiento real)."""

    ax, ay = cadera_medio
    bx, by = hombro_medio
    px, py = oreja

    largo_tronco = math.hypot(bx - ax, by - ay)

    if largo_tronco < 1e-6:
        return 0.0

    # Componente perpendicular del vector (oreja - cadera_medio)
    # respecto a la direccion del tronco (producto cruz en 2D).
    distancia_perpendicular = (
        (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    ) / largo_tronco

    return distancia_perpendicular / largo_tronco


def clasificar_postura(angulo_inclinacion, indice_joroba):

    etiquetas = []

    if abs(angulo_inclinacion) > UMBRAL_INCLINACION:
        lado = "adelante" if angulo_inclinacion > 0 else "atras"
        etiquetas.append(f"Inclinado {lado}")

    if abs(indice_joroba) > UMBRAL_JOROBA:
        etiquetas.append("Posible encorvamiento")

    if not etiquetas:
        etiquetas.append("Espalda recta")

    return " | ".join(etiquetas)


# =================================
# Programa principal
# =================================

def main():

    cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        raise RuntimeError(
            f"No pude abrir la camara con CAM_INDEX={CAM_INDEX}. "
            "Cierra otras aplicaciones que esten usando la camara."
        )

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=POSE_MODEL_PATH
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=USE_SEGMENTATION,
    )

    detector = vision.PoseLandmarker.create_from_options(options)

    print("Detector de pose + angulos iniciado.")
    print("Presiona 'q' para salir.")

    start_time = time.time()
    previous_time = time.time()

    while True:

        ok, frame_bgr = cap.read()

        if not ok:
            print("No se pudo recibir el frame de la camara.")
            break

        if ROTAR_FRAME is not None:
            frame_bgr = cv2.rotate(frame_bgr, ROTAR_FRAME)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

        timestamp_ms = int((time.time() - start_time) * 1000)

        result = detector.detect_for_video(mp_image, timestamp_ms)

        output = frame_bgr.copy()

        # =================================
        # SEGMENTACION
        # =================================

        if USE_SEGMENTATION and result.segmentation_masks:

            mask = result.segmentation_masks[0].numpy_view()
            mask = np.squeeze(mask)

            mask3 = np.repeat(mask[:, :, None], 3, axis=2)

            background_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            background_gray = cv2.cvtColor(
                background_gray, cv2.COLOR_GRAY2BGR
            )

            output = np.where(mask3 > MASK_THRESH, output, background_gray)

        # =================================
        # LANDMARKS + ANGULOS
        # =================================

        if result.pose_landmarks:

            pose_landmarks = result.pose_landmarks[0]
            height, width = output.shape[:2]

            points = [None] * len(pose_landmarks)

            for index, landmark in enumerate(pose_landmarks):

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                x = max(0, min(width - 1, x))
                y = max(0, min(height - 1, y))

                points[index] = (x, y)

                cv2.circle(output, (x, y), 3, (0, 255, 0), -1)

            for start, end in POSE_CONNECTIONS:
                if (
                    start < len(points)
                    and end < len(points)
                    and points[start] is not None
                    and points[end] is not None
                ):
                    cv2.line(output, points[start], points[end], (0, 255, 0), 2)

            # ---------------------------------
            # Angulo de rodilla y cadera (lado derecho)
            # ---------------------------------

            necesarios = [HOMBRO_D, CADERA_D, RODILLA_D, TOBILLO_D]

            if all(points[i] is not None for i in necesarios):

                angulo_rodilla = calcular_angulo(
                    points[CADERA_D], points[RODILLA_D], points[TOBILLO_D]
                )

                angulo_cadera = calcular_angulo(
                    points[HOMBRO_D], points[CADERA_D], points[RODILLA_D]
                )

                cv2.putText(
                    output, f"{angulo_rodilla:.0f}",
                    points[RODILLA_D], cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2,
                )

                cv2.putText(
                    output, f"{angulo_cadera:.0f}",
                    points[CADERA_D], cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2,
                )

            # ---------------------------------
            # Alineacion de espalda (hombro medio - cadera medio)
            # ---------------------------------

            necesarios_espalda = [HOMBRO_I, HOMBRO_D, CADERA_I, CADERA_D, OREJA_D]

            if all(points[i] is not None for i in necesarios_espalda):

                hombro_medio = punto_medio(points[HOMBRO_I], points[HOMBRO_D])
                cadera_medio = punto_medio(points[CADERA_I], points[CADERA_D])

                angulo_inclinacion = angulo_respecto_vertical(
                    cadera_medio, hombro_medio
                )

                indice_joroba = indice_encorvamiento(
                    points[OREJA_D], hombro_medio, cadera_medio
                )

                postura = clasificar_postura(angulo_inclinacion, indice_joroba)

                # Dibujar linea del eje de tronco
                cv2.line(
                    output,
                    (int(cadera_medio[0]), int(cadera_medio[1])),
                    (int(hombro_medio[0]), int(hombro_medio[1])),
                    (255, 0, 255), 2,
                )

                cv2.putText(
                    output,
                    f"Inclinacion: {angulo_inclinacion:.1f} grados",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 255), 2,
                )

                cv2.putText(
                    output,
                    f"Indice encorvamiento: {indice_joroba:.2f}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 255), 2,
                )

                cv2.putText(
                    output,
                    postura,
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2,
                )

        # =================================
        # FPS
        # =================================

        current_time = time.time()
        fps = 1.0 / max(1e-6, current_time - previous_time)
        previous_time = current_time

        cv2.putText(
            output, f"FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
        )

        alto_actual = output.shape[0]

        if alto_actual > ALTO_MAX_VENTANA:
            escala = ALTO_MAX_VENTANA / alto_actual
            nuevo_ancho = int(output.shape[1] * escala)
            output_mostrado = cv2.resize(output, (nuevo_ancho, ALTO_MAX_VENTANA))
        else:
            output_mostrado = output

        cv2.imshow("Pose + Angulos - q para salir", output_mostrado)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    print("Detector cerrado correctamente.")


if __name__ == "__main__":
    main()
