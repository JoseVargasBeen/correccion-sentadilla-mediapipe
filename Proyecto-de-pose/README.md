# Proyecto-de-pose

App de escritorio que analiza la sentadilla con MediaPipe Pose. Es el prototipo en el que se desarrolla y se mide la lógica antes de llevarla al celular. El resumen del proyecto completo está en el [README principal](../README.md).

## Ejecutar

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app_pose.py
```

Necesita `models/pose_landmarker_lite.task`, que se descarga desde la [página de Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker).

## Archivos

| Archivo | Qué hace |
|---|---|
| `app_pose.py` | La app: captura en un hilo, MediaPipe, medidas, conteo y panel |
| `conteo_sentadillas.py` | Máquina de estados que cuenta y juzga cada repetición |
| `perfil_corporal.py` | Proporciones de la persona, filtro de coherencia y la reconstrucción 3D (documentada como no usable con ruido real) |
| `calibrar_encorvamiento.py` | Mide tu índice de encorvamiento recto y encorvado, y propone el umbral |
| `conexion_qr.py` | Lee el QR de Camcel y conecta el PC a su red con `netsh` |
| `generar_icono.py` | Dibuja el ícono de las dos apps desde una sola fuente |
| `deteccion_aplausos.py` | Experimento: cambiar de persona activa con un aplauso |
| `bitacora_tecnica*.pdf` | Bitácoras técnicas del semillero |

## Pruebas

```bash
python -m doctest conteo_sentadillas.py app_pose.py
```
