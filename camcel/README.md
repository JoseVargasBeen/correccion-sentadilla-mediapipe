# Camcel

App Android que convierte el celular en una cámara de red para la [app de escritorio](../Proyecto-de-pose). Transmite video MJPEG en `http://<ip-del-celular>:8080/video`, el mismo formato que IP Webcam, así que cualquier programa que lea esa URL con `cv2.VideoCapture` funciona sin cambios.

## Qué tiene

- **Compresión nativa.** La cámara entrega NV21 y Kotlin lo comprime con `YuvImage.compressToJpeg`. La primera versión lo hacía en Dart, píxel a píxel, y el video iba a tirones.
- **Sin retraso acumulado.** Si el PC todavía no terminó de recibir el fotograma anterior, el siguiente se salta en vez de encolarse. Así el retraso no crece.
- **Punto de acceso y QR.** Con solo datos móviles, el PC no alcanza la IP del celular, porque queda detrás del NAT del operador. Camcel crea su propia red (`LocalOnlyHotspot`) y muestra un QR con la red, la clave y la dirección.
- **Selector de resolución** (240p, 480p, 720p) y contador de FPS, para medir el compromiso entre calidad y fluidez.
- **Pantalla encendida** mientras transmite: si se apaga, Android suspende la cámara.

## Archivos

| Archivo | Qué hace |
|---|---|
| `lib/main.dart` | Pantalla principal: vista previa, resolución, IP y botón de transmitir |
| `lib/mjpeg_server.dart` | Servidor HTTP y stream MJPEG con descarte de fotogramas |
| `lib/pantalla_qr.dart` | QR de conexión y punto de acceso |
| `lib/red.dart` | Búsqueda de la IP local y de la interfaz del punto de acceso |
| `lib/conversor_*.dart` | Conversión a JPEG: nativa, con respaldo en un isolate |
| `android/.../MainActivity.kt` | Compresor nativo, punto de acceso y pantalla encendida |

## Compilar

```bash
flutter build apk
flutter install
```

Esta carpeta no debe estar dentro de OneDrive ni de otra carpeta sincronizada: la sincronización reemplaza los archivos de compilación mientras Gradle los escribe y la compilación falla.
