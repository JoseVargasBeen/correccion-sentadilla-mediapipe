"""
pose_tasks_webcam.py

MediaPipe Pose Landmarker utilizando Tasks API.

Objetivo:
- Detectar una persona con la webcam.
- Obtener los 33 landmarks de pose.
- Dibujar puntos y conexiones tipo skeleton.
- Aplicar segmentación de persona/fondo.

Controles:
    q -> salir
"""

import time

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =================================
# Configuración
# =================================

# Fuente de video:
# - Webcam del PC: usar el índice de cámara, ej. 0
# - Celular con IP Webcam (Android, por WiFi): usar la URL del stream,
#   ej. "http://192.168.1.5:8080/video"
CAM_INDEX = 0
# CAM_INDEX = "http://192.168.1.5:8080/video"

# IP Webcam manda el video con la orientacion fija de cuando arranco
# la app (no gira solo si rotas el celular). Si la imagen sale de
# lado o al reves, forzar la rotacion aqui:
#   None                        -> no rotar
#   cv2.ROTATE_90_CLOCKWISE     -> celular en vertical, cable a la izq.
#   cv2.ROTATE_90_COUNTERCLOCKWISE -> celular en vertical, cable a la der.
#   cv2.ROTATE_180              -> imagen al reves
ROTAR_FRAME = None

POSE_MODEL_PATH = "models/pose_landmarker_lite.task"

# Activar segmentación
USE_SEGMENTATION = True

# Umbral de la máscara.
# Un valor mayor hace la máscara más estricta.
MASK_THRESH = 0.2


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
# Programa principal
# =================================

def main():

    # ---------------------------------
    # Abrir cámara
    # ---------------------------------

    cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():

        raise RuntimeError(
            f"No pude abrir la cámara con CAM_INDEX={CAM_INDEX}. "
            "Cierra otras aplicaciones que estén usando la cámara."
        )


    # ---------------------------------
    # Configurar Pose Landmarker
    # ---------------------------------

    options = vision.PoseLandmarkerOptions(

        base_options=python.BaseOptions(
            model_asset_path=POSE_MODEL_PATH
        ),

        # Webcam = modo VIDEO
        running_mode=vision.RunningMode.VIDEO,

        # Detectar una persona
        num_poses=1,

        # Confianza mínima para detectar pose
        min_pose_detection_confidence=0.5,

        # Confianza mínima de presencia
        min_pose_presence_confidence=0.5,

        # Confianza mínima del tracking
        min_tracking_confidence=0.5,

        # Activar máscara persona/fondo
        output_segmentation_masks=USE_SEGMENTATION,
    )


    # ---------------------------------
    # Crear detector
    # ---------------------------------

    detector = vision.PoseLandmarker.create_from_options(
        options
    )


    print("Detector de pose iniciado.")
    print("Presiona 'q' para salir.")


    # ---------------------------------
    # Variables de tiempo
    # ---------------------------------

    start_time = time.time()
    previous_time = time.time()


    # =================================
    # Bucle principal
    # =================================

    while True:

        # ---------------------------------
        # Leer frame
        # ---------------------------------

        ok, frame_bgr = cap.read()

        if not ok:

            print(
                "No se pudo recibir el frame de la cámara."
            )

            break


        if ROTAR_FRAME is not None:
            frame_bgr = cv2.rotate(frame_bgr, ROTAR_FRAME)


        # ---------------------------------
        # BGR -> RGB
        # ---------------------------------

        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB
        )


        # ---------------------------------
        # Crear imagen MediaPipe
        # ---------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )


        # ---------------------------------
        # Timestamp en milisegundos
        # ---------------------------------

        timestamp_ms = int(
            (time.time() - start_time) * 1000
        )


        # ---------------------------------
        # Detectar pose
        # ---------------------------------

        result = detector.detect_for_video(
            mp_image,
            timestamp_ms
        )


        # ---------------------------------
        # Crear imagen de salida
        # ---------------------------------

        output = frame_bgr.copy()


        # =================================
        # SEGMENTACIÓN
        # =================================

        if (
            USE_SEGMENTATION
            and result.segmentation_masks
        ):

            # Obtener máscara
            mask = result.segmentation_masks[
                0
            ].numpy_view()


            # Asegurar formato H x W
            mask = np.squeeze(mask)


            # Convertir máscara a 3 canales
            mask3 = np.repeat(
                mask[:, :, None],
                3,
                axis=2
            )


            # Crear fondo en escala de grises
            background_gray = cv2.cvtColor(
                output,
                cv2.COLOR_BGR2GRAY
            )

            background_gray = cv2.cvtColor(
                background_gray,
                cv2.COLOR_GRAY2BGR
            )


            # Mantener persona a color
            # y convertir fondo a gris
            output = np.where(
                mask3 > MASK_THRESH,
                output,
                background_gray
            )


        # =================================
        # LANDMARKS + SKELETON
        # =================================

        if result.pose_landmarks:

            # Tomar la primera persona
            pose_landmarks = (
                result.pose_landmarks[0]
            )

            height, width = output.shape[:2]


            # ---------------------------------
            # Convertir landmarks a píxeles
            # ---------------------------------

            points = [
                None
            ] * len(pose_landmarks)


            for index, landmark in enumerate(
                pose_landmarks
            ):

                # Coordenada X
                x = int(
                    landmark.x * width
                )

                # Coordenada Y
                y = int(
                    landmark.y * height
                )


                # Mantener dentro de la imagen
                x = max(
                    0,
                    min(width - 1, x)
                )

                y = max(
                    0,
                    min(height - 1, y)
                )


                points[index] = (
                    x,
                    y
                )


                # Dibujar landmark
                cv2.circle(
                    output,
                    (x, y),
                    3,
                    (0, 255, 0),
                    -1
                )


            # ---------------------------------
            # Dibujar conexiones
            # ---------------------------------

            for start, end in POSE_CONNECTIONS:

                if (
                    start < len(points)
                    and end < len(points)
                    and points[start] is not None
                    and points[end] is not None
                ):

                    cv2.line(
                        output,
                        points[start],
                        points[end],
                        (0, 255, 0),
                        2
                    )


        # =================================
        # CALCULAR FPS
        # =================================

        current_time = time.time()

        fps = 1.0 / max(
            1e-6,
            current_time - previous_time
        )

        previous_time = current_time


        # ---------------------------------
        # Mostrar FPS
        # ---------------------------------

        cv2.putText(
            output,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )


        # ---------------------------------
        # Mostrar ventana
        # ---------------------------------

        cv2.imshow(
            "MediaPipe Pose - q para salir",
            output
        )


        # ---------------------------------
        # Salir con q
        # ---------------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


    # =================================
    # Liberar recursos
    # =================================

    cap.release()

    cv2.destroyAllWindows()

    detector.close()

    print(
        "Detector de pose cerrado correctamente."
    )


# =================================
# Ejecutar programa
# =================================

if __name__ == "__main__":
    main()
