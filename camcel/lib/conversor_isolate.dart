import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'conversor_frame.dart';

/// Un isolate en Dart es como un hilo, pero sin memoria compartida: los
/// datos viajan por mensajes. Se usa aqui para que convertir cada frame
/// (YUV -> RGB -> JPEG, que es trabajo pesado de CPU) no bloquee el
/// isolate principal, que es el que dibuja la interfaz y el preview de
/// la camara. Sin esto, la app se siente entrecortada mientras transmite.
///
/// Se mantiene UN isolate vivo mientras dure la transmision, en vez de
/// crear uno por frame (crear isolates tiene costo).
class ConversorIsolate {
  Isolate? _isolate;
  SendPort? _puertoTrabajo;

  bool get estaListo => _puertoTrabajo != null;

  Future<void> iniciar() async {
    if (_isolate != null) return;

    final puertoArranque = ReceivePort();

    _isolate = await Isolate.spawn(_bucleDeTrabajo, puertoArranque.sendPort);

    _puertoTrabajo = await puertoArranque.first as SendPort;
    puertoArranque.close();
  }

  Future<Uint8List?> convertir(DatosFrameYuv datos) async {
    final puerto = _puertoTrabajo;
    if (puerto == null) return null;

    final puertoRespuesta = ReceivePort();

    puerto.send(_Peticion(puertoRespuesta.sendPort, datos));

    final resultado = await puertoRespuesta.first;
    puertoRespuesta.close();

    return resultado as Uint8List?;
  }

  void detener() {
    _isolate?.kill(priority: Isolate.immediate);
    _isolate = null;
    _puertoTrabajo = null;
  }
}

class _Peticion {
  final SendPort puertoRespuesta;
  final DatosFrameYuv datos;

  const _Peticion(this.puertoRespuesta, this.datos);
}

/// Este codigo corre YA dentro del isolate secundario.
void _bucleDeTrabajo(SendPort puertoPrincipal) {
  final puertoEntrada = ReceivePort();

  // Primer mensaje: devolverle al isolate principal por donde mandarnos
  // trabajo.
  puertoPrincipal.send(puertoEntrada.sendPort);

  puertoEntrada.listen((mensaje) {
    final peticion = mensaje as _Peticion;

    try {
      peticion.puertoRespuesta.send(convertirYuvAJpeg(peticion.datos));
    } catch (_) {
      peticion.puertoRespuesta.send(null);
    }
  });
}
