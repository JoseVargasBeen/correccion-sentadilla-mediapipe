import 'dart:typed_data';

import 'package:image/image.dart' as img;

/// Datos crudos de un frame, en formato plano y "enviable" entre
/// isolates. Un CameraImage no se puede mandar directo a otro isolate
/// porque envuelve objetos de plataforma, por eso se extraen los bytes.
class DatosFrameYuv {
  final Uint8List planoY;
  final Uint8List planoU;
  final Uint8List planoV;

  final int ancho;
  final int alto;

  final int strideY;
  final int strideUv;
  final int pixelStrideUv;

  final int calidad;

  const DatosFrameYuv({
    required this.planoY,
    required this.planoU,
    required this.planoV,
    required this.ancho,
    required this.alto,
    required this.strideY,
    required this.strideUv,
    required this.pixelStrideUv,
    required this.calidad,
  });
}

/// La camara de Android entrega cada frame en formato YUV420 (luminancia
/// + color, no RGB directo). Para mandarlo como JPEG por HTTP hay que
/// convertirlo primero a RGB.
///
/// Usa aritmetica entera de punto fijo y escribe directo a un buffer de
/// bytes, en vez de llamar setPixelRgb por pixel (que tiene overhead de
/// objeto). Sin esto, la conversion se vuelve el cuello de botella y el
/// video se ve entrecortado.
Uint8List convertirYuvAJpeg(DatosFrameYuv datos) {
  final ancho = datos.ancho;
  final alto = datos.alto;

  final bufferY = datos.planoY;
  final bufferU = datos.planoU;
  final bufferV = datos.planoV;

  final rgb = Uint8List(ancho * alto * 3);

  int posicionRgb = 0;

  for (int y = 0; y < alto; y++) {
    final filaY = y * datos.strideY;
    final filaUv = (y >> 1) * datos.strideUv;

    for (int x = 0; x < ancho; x++) {
      final valorY = bufferY[filaY + x];
      final indiceUv = filaUv + (x >> 1) * datos.pixelStrideUv;

      final u = bufferU[indiceUv] - 128;
      final v = bufferV[indiceUv] - 128;

      // BT.601 en punto fijo: todo enteros, >> 10 equivale a dividir
      // por 1024.
      final y1192 = valorY * 1192;

      final r = (y1192 + 1634 * v) >> 10;
      final g = (y1192 - 833 * v - 400 * u) >> 10;
      final b = (y1192 + 2066 * u) >> 10;

      rgb[posicionRgb] = r < 0 ? 0 : (r > 255 ? 255 : r);
      rgb[posicionRgb + 1] = g < 0 ? 0 : (g > 255 ? 255 : g);
      rgb[posicionRgb + 2] = b < 0 ? 0 : (b > 255 ? 255 : b);

      posicionRgb += 3;
    }
  }

  final imagen = img.Image.fromBytes(
    width: ancho,
    height: alto,
    bytes: rgb.buffer,
    numChannels: 3,
    order: img.ChannelOrder.rgb,
  );

  return Uint8List.fromList(img.encodeJpg(imagen, quality: datos.calidad));
}
