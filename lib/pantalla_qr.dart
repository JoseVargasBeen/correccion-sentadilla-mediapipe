import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:qr_flutter/qr_flutter.dart';

import 'conversor_nativo.dart';
import 'red.dart';
import 'tema.dart';

/// Pantalla que muestra un QR con todo lo que el PC necesita para
/// conectarse, para no tener que teclear direcciones a mano.
///
/// El QR lleva un JSON con dos formas:
///
/// - Misma red (WiFi de casa o anclaje ya encendido):
///   {"ip": "192.168.1.14", "puerto": 8080}
///   El PC solo se conecta a esa direccion.
///
/// - Punto de acceso del celular:
///   {"ip": "...", "puerto": 8080, "ssid": "...", "clave": "..."}
///   El PC primero entra a esa red y despues se conecta.
class PantallaQr extends StatefulWidget {
  const PantallaQr({
    super.key,
    required this.direccionIp,
    required this.puerto,
  });

  final String? direccionIp;
  final int puerto;

  @override
  State<PantallaQr> createState() => _PantallaQrState();
}

class _PantallaQrState extends State<PantallaQr> {
  bool _ocupado = false;
  String? _ssid;
  String? _clave;
  String? _error;

  /// La IP cambia al encender el punto de acceso: el celular deja la
  /// WiFi y pasa a ser el router de su propia red.
  String? _ipPuntoDeAcceso;

  bool get _puntoDeAccesoActivo => _ssid != null;

  String? get _ipActual => _ipPuntoDeAcceso ?? widget.direccionIp;

  String? get _contenidoQr {
    final ip = _ipActual;
    if (ip == null) return null;

    final datos = <String, dynamic>{'ip': ip, 'puerto': widget.puerto};

    if (_puntoDeAccesoActivo) {
      datos['ssid'] = _ssid;
      datos['clave'] = _clave;
    }

    return jsonEncode(datos);
  }

  Future<void> _alternarPuntoDeAcceso() async {
    setState(() {
      _ocupado = true;
      _error = null;
    });

    try {
      if (_puntoDeAccesoActivo) {
        await ConversorNativo.cerrarPuntoDeAcceso();

        if (!mounted) return;
        setState(() {
          _ssid = null;
          _clave = null;
          _ipPuntoDeAcceso = null;
        });
      } else {
        // Foto de las interfaces ANTES de crear la red, para despues
        // reconocer cual apareció.
        final previas = await direccionesPorInterfaz();

        final credenciales = await ConversorNativo.abrirPuntoDeAcceso();
        final ip = await buscarIpNueva(previas);

        if (!mounted) return;

        if (ip == null) {
          await ConversorNativo.cerrarPuntoDeAcceso();
          setState(() {
            _error =
                'La red se creo pero no aparecio su direccion. '
                'Apaga el anclaje si lo tienes encendido y reintenta.';
          });
          return;
        }

        setState(() {
          _ssid = credenciales.ssid;
          _clave = credenciales.clave;
          _ipPuntoDeAcceso = ip;
        });
      }
    } on PlatformException catch (e) {
      if (!mounted) return;
      setState(() => _error = e.message ?? 'No se pudo crear la red');
    } finally {
      if (mounted) setState(() => _ocupado = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final contenido = _contenidoQr;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Conectar por QR'),
        backgroundColor: Paleta.superficie,
        foregroundColor: Paleta.texto,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(Espacio.lg),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              if (contenido != null)
                _recuadroQr(contenido)
              else
                _aviso(
                  'Sin direccion todavia. Conectate a una WiFi o enciende '
                  'el punto de acceso aqui abajo.',
                ),
              const SizedBox(height: Espacio.lg),
              Fila('Direccion', valor: _ipActual),
              Fila('Red', valor: _puntoDeAccesoActivo ? _ssid : 'WiFi actual'),
              if (_puntoDeAccesoActivo) Fila('Clave', valor: _clave),
              const SizedBox(height: Espacio.lg),
              _explicacion(context),
              const SizedBox(height: Espacio.lg),
              if (_error != null) ...[
                _aviso(_error!, alerta: true),
                const SizedBox(height: Espacio.md),
              ],
              SizedBox(
                height: 46,
                child: FilledButton(
                  onPressed: _ocupado ? null : _alternarPuntoDeAcceso,
                  style: FilledButton.styleFrom(
                    backgroundColor: _puntoDeAccesoActivo
                        ? Paleta.campo
                        : Paleta.texto,
                    foregroundColor: _puntoDeAccesoActivo
                        ? Paleta.texto
                        : Paleta.superficie,
                    disabledBackgroundColor: Paleta.campo,
                    disabledForegroundColor: Paleta.filete,
                    shape: const RoundedRectangleBorder(),
                    textStyle: Theme.of(context).textTheme.labelLarge,
                  ),
                  child: Text(
                    _ocupado
                        ? 'Un momento...'
                        : _puntoDeAccesoActivo
                        ? 'Apagar punto de acceso'
                        : 'Crear punto de acceso',
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _recuadroQr(String contenido) {
    return Center(
      child: Container(
        padding: const EdgeInsets.all(Espacio.md),
        // El QR necesita fondo claro y modulos oscuros para que
        // cualquier lector lo tome sin pelear con el contraste.
        color: Paleta.texto,
        child: QrImageView(
          data: contenido,
          version: QrVersions.auto,
          size: 240,
          backgroundColor: Paleta.texto,
          eyeStyle: const QrEyeStyle(
            eyeShape: QrEyeShape.square,
            color: Paleta.lienzo,
          ),
          dataModuleStyle: const QrDataModuleStyle(
            dataModuleShape: QrDataModuleShape.square,
            color: Paleta.lienzo,
          ),
        ),
      ),
    );
  }

  Widget _aviso(String texto, {bool alerta = false}) {
    return Container(
      padding: const EdgeInsets.all(Espacio.md),
      decoration: BoxDecoration(
        color: Paleta.campo,
        border: Border.all(color: alerta ? Paleta.alerta : Paleta.filete),
      ),
      child: Text(
        texto,
        style: TextStyle(color: alerta ? Paleta.texto : Paleta.tenue),
      ),
    );
  }

  Widget _explicacion(BuildContext context) {
    final texto = _puntoDeAccesoActivo
        ? 'El celular creo su propia red. Escanea el QR desde el PC y se '
              'conectara solo. Esta red no comparte internet: sirve para '
              'que el PC vea el video, nada mas.'
        : 'Si el PC y el celular ya estan en la misma WiFi, escanea el QR '
              'y listo. Si solo tienes datos moviles, crea el punto de '
              'acceso aqui abajo.';

    return Text(texto, style: Theme.of(context).textTheme.labelSmall);
  }
}
