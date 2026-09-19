"""
deteccion_aplausos.py

Prototipo: detectar varias personas a la vez y rotar cual de ellas
esta "activa" (la que se analiza) cuando alguien aplaude.

Como funciona:
- Pose Landmarker corre con num_poses > 1, asi que detecta a todas
  las personas visibles (cada una con sus 33 landmarks).
- Hand Landmarker corre en paralelo detectando manos de cualquiera
  en el cuadro.
- Un "aplauso" se detecta cuando dos manos (de quien sea) se juntan
  por debajo de un umbral de distancia. Al detectarse, la persona
  activa cambia a la siguiente (rotando en orden).
- La persona activa se dibuja en amarillo; las demas en verde tenue.

Limitacion conocida: MediaPipe Pose (Tasks API) no asigna un ID
estable por persona entre frames -- el orden de result.pose_landmarks
puede cambiar si la gente se mueve mucho o se cruza. Para este
prototipo (pocas personas, movimiento moderado) es aceptable; si
se necesita tracking robusto de identidad habria que agregar
re-identificacion (fuera del alcance de este prototipo).

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

CAM_INDEX = 0
# CAM_INDEX = "http://192.168.1.8:8080/video"

ROTAR_FRAME = None  # cv2.ROTATE_90_CLOCKWISE, etc. Ver otros scripts.

POSE_MODEL_PATH = "models/pose_landmarker_lite.task"
HAND_MODEL_PATH = "models/hand_landmarker.task"

NUM_PERSONAS_MAX = 4
NUM_MANOS_MAX = 8  # hasta 4 personas x 2 manos

# Distancia (en px) por debajo de la cual dos manos se consideran
# "juntas". Depende de la resolucion y distancia a la camara; ajustar
# probando.
UMBRAL_DISTANCIA_APLAUSO = 70

# Tiempo minimo (segundos) entre un cambio de persona y el siguiente,
# para no rotar varias veces por el mismo aplauso.
COOLDOWN_SEGUNDOS = 1.5

ALTO_MAX_VENTANA = 900


# =================================
# Conexiones del skeleton
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
# Deteccion de aplauso
# =================================

def centroide_mano(hand_landmarks, width, height):
    xs = [lm.x * width for lm in hand_landmarks]
    ys = [lm.y * height for lm in hand_landmarks]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def hay_manos_juntas(centroides, umbral):
    """True si algun par de manos esta a menos de `umbral` px."""

    for i in range(len(centroides)):
        for j in range(i + 1, len(centroides)):
            dx = centroides[i][0] - centroides[j][0]
            dy = centroides[i][1] - centroides[j][1]
            distancia = math.hypot(dx, dy)

            if distancia < umbral:
                return True

    return False


# =================================
# Programa principal
# =================================

def main():

    cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        raise RuntimeError(
            f"No pude abrir la camara con CAM_INDEX={CAM_INDEX}."
        )

    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=NUM_PERSONAS_MAX,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=NUM_MANOS_MAX,
    )
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)

    print("Deteccion de aplausos iniciada.")
    print("Presiona 'q' para salir.")

    start_time = time.time()
    previous_time = time.time()

    persona_activa = 0
    ultimo_cambio = 0.0
    mensaje_cambio = ""
    mensaje_hasta = 0.0

    while True:

        ok, frame_bgr = cap.read()

        if not ok:
            print("No se pudo recibir el frame de la camara.")
            break

        if ROTAR_FRAME is not None:
            frame_bgr = cv2.rotate(frame_bgr, ROTAR_FRAME)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp_ms = int((time.time() - start_time) * 1000)

        pose_result = pose_detector.detect_for_video(mp_image, timestamp_ms)
        hand_result = hand_detector.detect_for_video(mp_image, timestamp_ms)

        output = frame_bgr.copy()
        height, width = output.shape[:2]

        # =================================
        # Dibujar todas las personas detectadas
        # =================================

        num_personas = len(pose_result.pose_landmarks)

        # Si la persona activa ya no existe (alguien se fue), acotar.
        if num_personas > 0:
            persona_activa = persona_activa % num_personas

        for idx_persona, pose_landmarks in enumerate(pose_result.pose_landmarks):

            es_activa = idx_persona == persona_activa
            color = (0, 255, 255) if es_activa else (0, 150, 0)

            points = [None] * len(pose_landmarks)

            for i, landmark in enumerate(pose_landmarks):
                x = max(0, min(width - 1, int(landmark.x * width)))
                y = max(0, min(height - 1, int(landmark.y * height)))
                points[i] = (x, y)
                cv2.circle(output, (x, y), 3, color, -1)

            for start, end in POSE_CONNECTIONS:
                if (
                    start < len(points) and end < len(points)
                    and points[start] is not None and points[end] is not None
                ):
                    cv2.line(output, points[start], points[end], color, 2)

            if es_activa and points[0] is not None:
                cv2.putText(
                    output, "ACTIVA", (points[0][0] - 30, points[0][1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                )

        # =================================
        # Deteccion de aplauso (manos juntas)
        # =================================

        centroides = [
            centroide_mano(mano, width, height)
            for mano in hand_result.hand_landmarks
        ]

        for cx, cy in centroides:
            cv2.circle(output, (int(cx), int(cy)), 8, (255, 0, 255), 2)

        ahora = time.time()
        en_cooldown = (ahora - ultimo_cambio) < COOLDOWN_SEGUNDOS

        if (
            len(centroides) >= 2
            and not en_cooldown
            and hay_manos_juntas(centroides, UMBRAL_DISTANCIA_APLAUSO)
            and num_personas > 0
        ):
            persona_activa = (persona_activa + 1) % num_personas
            ultimo_cambio = ahora
            mensaje_cambio = f"Aplauso detectado -> persona activa: {persona_activa}"
            mensaje_hasta = ahora + 2.0
            print(mensaje_cambio)

        # =================================
        # Overlay de estado
        # =================================

        cv2.putText(
            output, f"Personas detectadas: {num_personas}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )

        cv2.putText(
            output, f"Persona activa: {persona_activa}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
        )

        if ahora < mensaje_hasta:
            cv2.putText(
                output, mensaje_cambio, (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
            )

        current_time = time.time()
        fps = 1.0 / max(1e-6, current_time - previous_time)
        previous_time = current_time

        cv2.putText(
            output, f"FPS: {fps:.1f}", (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

        alto_actual = output.shape[0]

        if alto_actual > ALTO_MAX_VENTANA:
            escala = ALTO_MAX_VENTANA / alto_actual
            nuevo_ancho = int(output.shape[1] * escala)
            output_mostrado = cv2.resize(output, (nuevo_ancho, ALTO_MAX_VENTANA))
        else:
            output_mostrado = output

        cv2.imshow("Deteccion de aplausos - q para salir", output_mostrado)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose_detector.close()
    hand_detector.close()

    print("Detector cerrado correctamente.")


if __name__ == "__main__":
    main()
