import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

/// Servidor HTTP que expone el video de la camara como un stream MJPEG
/// en la ruta /video, igual que hace la app IP Webcam.
///
/// Un MJPEG es, en el fondo, una respuesta HTTP que nunca termina: en vez
/// de mandar un solo archivo, manda "imagen JPEG, imagen JPEG, imagen
/// JPEG..." una tras otra separadas por un marcador (`boundary`). El
/// navegador (o cv2.VideoCapture del lado de Python) lo interpreta como
/// video en vivo.
class MjpegServer {
  final int puerto;

  HttpServer? _servidor;
  final List<_ClienteStream> _clientesConectados = [];

  MjpegServer({this.puerto = 8080});

  bool get estaCorriendo => _servidor != null;

  Future<void> iniciar() async {
    _servidor = await HttpServer.bind(InternetAddress.anyIPv4, puerto);
    _servidor!.listen(_manejarPeticion);
  }

  void _manejarPeticion(HttpRequest peticion) async {
    if (peticion.uri.path != '/video') {
      peticion.response.statusCode = HttpStatus.notFound;
      await peticion.response.close();
      return;
    }

    // Encabezados que le dicen al cliente "esto es un stream MJPEG".
    peticion.response.headers.set(
      HttpHeaders.contentTypeHeader,
      'multipart/x-mixed-replace; boundary=camcelframe',
    );
    peticion.response.headers.set(HttpHeaders.cacheControlHeader, 'no-cache');

    final cliente = _ClienteStream(peticion.response);
    _clientesConectados.add(cliente);

    // Cuando el cliente se desconecta (cierra la app, pierde la red),
    // lo sacamos de la lista para no seguir intentando escribirle.
    unawaited(
      peticion.response.done.catchError((_) {}).whenComplete(() {
        _clientesConectados.remove(cliente);
      }),
    );
  }

  /// Llamar esto por cada frame nuevo ya codificado en JPEG.
  ///
  /// Si un cliente todavia no termino de recibir el frame anterior, este
  /// se le SALTA en vez de encolarselo. Encolar hace que el retraso
  /// crezca sin limite: el celular seguiria mandando frames viejos
  /// acumulados mientras la realidad ya avanzo.
  void publicarFrame(Uint8List jpegBytes) {
    if (_clientesConectados.isEmpty) return;

    final encabezadoFrame = ascii.encode(
      '--camcelframe\r\n'
      'Content-Type: image/jpeg\r\n'
      'Content-Length: ${jpegBytes.length}\r\n\r\n',
    );

    final cierre = ascii.encode('\r\n');

    for (final cliente in List<_ClienteStream>.of(_clientesConectados)) {
      if (cliente.enviando) continue;

      cliente.enviando = true;

      try {
        cliente.respuesta
          ..add(encabezadoFrame)
          ..add(jpegBytes)
          ..add(cierre);

        cliente.respuesta
            .flush()
            .then((_) {
              cliente.enviando = false;
            })
            .onError((_, _) {
              cliente.enviando = false;
              _clientesConectados.remove(cliente);
            });
      } catch (_) {
        cliente.enviando = false;
        _clientesConectados.remove(cliente);
      }
    }
  }

  Future<void> detener() async {
    for (final cliente in _clientesConectados) {
      await cliente.respuesta.close().catchError((_) {});
    }
    _clientesConectados.clear();

    await _servidor?.close(force: true);
    _servidor = null;
  }
}

class _ClienteStream {
  final HttpResponse respuesta;

  /// true mientras el frame anterior aun no termina de salir por el
  /// socket. Sirve para saltarse frames en vez de acumularlos.
  bool enviando = false;

  _ClienteStream(this.respuesta);
}
