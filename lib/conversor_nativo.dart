import 'package:flutter/services.dart';

/// Llama al codigo Kotlin de MainActivity para comprimir un frame NV21
/// a JPEG usando YuvImage de Android (nativo, mucho mas rapido que
/// hacerlo pixel por pixel en Dart).
///
/// Un MethodChannel es el mecanismo estandar de Flutter para hablar con
/// codigo nativo: se manda un nombre de metodo + argumentos, y llega la
/// respuesta de vuelta. El nombre del canal debe coincidir exactamente
/// con el declarado en MainActivity.kt.
class ConversorNativo {
  static const MethodChannel _canal = MethodChannel('camcel/jpeg');

  static bool disponible = true;

  static Future<Uint8List?> nv21AJpeg({
    required Uint8List nv21,
    required int ancho,
    required int alto,
    int calidad = 75,
  }) async {
    if (!disponible) return null;

    try {
      return await _canal.invokeMethod<Uint8List>('nv21AJpeg', {
        'bytes': nv21,
        'ancho': ancho,
        'alto': alto,
        'calidad': calidad,
      });
    } on MissingPluginException {
      // No hay implementacion nativa (ej. corriendo en otra plataforma):
      // se marca como no disponible y el resto del codigo usa el
      // conversor en Dart como respaldo.
      disponible = false;
      return null;
    } catch (_) {
      return null;
    }
  }

  /// Mantiene la pantalla encendida mientras se transmite. Si se apaga,
  /// Android suspende la camara y se corta el stream.
  static Future<void> mantenerPantallaEncendida(bool activar) async {
    try {
      await _canal.invokeMethod('mantenerPantallaEncendida', {
        'activar': activar,
      });
    } catch (_) {
      // Si no hay implementacion nativa, se sigue sin esta mejora.
    }
  }

  /// Crea la red WiFi propia del celular y devuelve sus credenciales,
  /// que son las que viajan en el QR. Lanza [PlatformException] con un
  /// mensaje legible si Android se niega.
  static Future<({String ssid, String clave})> abrirPuntoDeAcceso() async {
    final datos = await _canal.invokeMapMethod<String, String>(
      'abrirPuntoDeAcceso',
    );

    if (datos == null) {
      throw PlatformException(
        code: 'SIN_DATOS',
        message: 'Android no devolvio las credenciales de la red',
      );
    }

    return (ssid: datos['ssid'] ?? '', clave: datos['clave'] ?? '');
  }

  static Future<void> cerrarPuntoDeAcceso() async {
    try {
      await _canal.invokeMethod('cerrarPuntoDeAcceso');
    } catch (_) {
      // Si ya estaba cerrado no hay nada que hacer.
    }
  }
}
