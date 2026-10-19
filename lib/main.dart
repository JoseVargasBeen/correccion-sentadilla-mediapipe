import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import 'conversor_frame.dart';
import 'conversor_isolate.dart';
import 'conversor_nativo.dart';
import 'mjpeg_server.dart';
import 'pantalla_qr.dart';
import 'red.dart';
import 'tema.dart';

const int puertoServidor = 8080;

const Map<String, ResolutionPreset> resolucionesDisponibles = {
  'Baja 240p': ResolutionPreset.low,
  'Media 480p': ResolutionPreset.medium,
  'Alta 720p': ResolutionPreset.high,
};

void main() {
  runApp(const CamcelApp());
}

class CamcelApp extends StatelessWidget {
  const CamcelApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Camcel',
      debugShowCheckedModeBanner: false,
      theme: construirTema(),
      home: const PantallaCamara(),
    );
  }
}

class PantallaCamara extends StatefulWidget {
  const PantallaCamara({super.key});

  @override
  State<PantallaCamara> createState() => _PantallaCamaraState();
}

class _PantallaCamaraState extends State<PantallaCamara> {
  CameraController? _controlador;
  final MjpegServer _servidor = MjpegServer(puerto: puertoServidor);
  final ConversorIsolate _conversor = ConversorIsolate();

  bool _transmitiendo = false;
  bool _procesandoFrame = false;
  String? _direccionIp;
  String _estado = 'Inicializando camara...';
  String _resolucionSeleccionada = 'Media 480p';

  int _framesEnviados = 0;
  DateTime _ultimaMedicion = DateTime.now();
  double _fps = 0;
  bool _usandoConversorNativo = false;

  @override
  void initState() {
    super.initState();
    _inicializarCamara();
    _buscarIpLocal();
  }

  Future<void> _inicializarCamara() async {
    try {
      final camaras = await availableCameras();

      final camaraTrasera = camaras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => camaras.first,
      );

      final controlador = CameraController(
        camaraTrasera,
        resolucionesDisponibles[_resolucionSeleccionada]!,
        enableAudio: false,
        // NV21 llega como un solo buffer, que es justo lo que espera
        // YuvImage en Android para comprimir a JPEG de forma nativa.
        imageFormatGroup: ImageFormatGroup.nv21,
      );

      await controlador.initialize();

      if (!mounted) return;

      setState(() {
        _controlador = controlador;
        _estado = 'Camara lista';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _estado = 'Error abriendo camara: $e');
    }
  }

  Future<void> _cambiarResolucion(String nueva) async {
    if (nueva == _resolucionSeleccionada) return;

    final estabaTransmitiendo = _transmitiendo;

    if (estabaTransmitiendo) await _detenerTransmision();

    await _controlador?.dispose();

    if (!mounted) return;

    setState(() {
      _controlador = null;
      _resolucionSeleccionada = nueva;
      _estado = 'Cambiando resolucion...';
    });

    await _inicializarCamara();

    if (estabaTransmitiendo) await _iniciarTransmision();
  }

  Future<void> _buscarIpLocal() async {
    final ip = await buscarIpLocal();

    if (!mounted || ip == null) return;

    setState(() => _direccionIp = ip);
  }

  Future<void> _abrirPantallaQr() async {
    await Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => PantallaQr(
          direccionIp: _direccionIp,
          puerto: puertoServidor,
        ),
      ),
    );

    // Al volver, la red puede haber cambiado: crear el punto de acceso
    // tumba la WiFi y con ella la direccion que se mostraba.
    await _buscarIpLocal();
  }

  Future<void> _alternarTransmision() async {
    if (_transmitiendo) {
      await _detenerTransmision();
    } else {
      await _iniciarTransmision();
    }
  }

  Future<void> _iniciarTransmision() async {
    final controlador = _controlador;

    if (controlador == null || !controlador.value.isInitialized) return;

    await _conversor.iniciar();
    await _servidor.iniciar();
    await ConversorNativo.mantenerPantallaEncendida(true);

    await controlador.startImageStream((CameraImage frame) {
      // Si el frame anterior todavia se esta convirtiendo, este se
      // descarta. Evita que se acumulen frames y crezca la latencia.
      if (_procesandoFrame) return;

      _procesandoFrame = true;

      _procesarYPublicar(frame).whenComplete(() {
        _procesandoFrame = false;
      });
    });

    if (!mounted) return;

    setState(() {
      _transmitiendo = true;
      _estado = 'Transmitiendo';
      _framesEnviados = 0;
      _ultimaMedicion = DateTime.now();
    });
  }

  Future<void> _procesarYPublicar(CameraImage frame) async {
    try {
      final jpegBytes = await _convertirAJpeg(frame);

      if (jpegBytes == null) return;

      _servidor.publicarFrame(jpegBytes);
      _contarFrame();
    } catch (_) {
      // Si un frame falla, simplemente se salta.
    }
  }

  /// Ruta rapida: si la camara entrego NV21 (un solo plano) y el codigo
  /// nativo esta disponible, se comprime en Android. Si no, se usa el
  /// conversor en Dart dentro del isolate como respaldo.
  Future<Uint8List?> _convertirAJpeg(CameraImage frame) async {
    if (frame.planes.length == 1 && ConversorNativo.disponible) {
      final jpeg = await ConversorNativo.nv21AJpeg(
        nv21: frame.planes[0].bytes,
        ancho: frame.width,
        alto: frame.height,
        calidad: 75,
      );

      if (jpeg != null) {
        if (!_usandoConversorNativo) {
          _usandoConversorNativo = true;
          if (mounted) setState(() {});
        }
        return jpeg;
      }
    }

    if (frame.planes.length < 3) return null;

    _usandoConversorNativo = false;

    return _conversor.convertir(
      DatosFrameYuv(
        planoY: frame.planes[0].bytes,
        planoU: frame.planes[1].bytes,
        planoV: frame.planes[2].bytes,
        ancho: frame.width,
        alto: frame.height,
        strideY: frame.planes[0].bytesPerRow,
        strideUv: frame.planes[1].bytesPerRow,
        pixelStrideUv: frame.planes[1].bytesPerPixel ?? 1,
        calidad: 75,
      ),
    );
  }

  void _contarFrame() {
    _framesEnviados++;

    final ahora = DateTime.now();
    final transcurrido = ahora.difference(_ultimaMedicion);

    if (transcurrido.inMilliseconds >= 1000) {
      final fps = _framesEnviados * 1000 / transcurrido.inMilliseconds;

      _framesEnviados = 0;
      _ultimaMedicion = ahora;

      if (mounted) setState(() => _fps = fps);
    }
  }

  Future<void> _detenerTransmision() async {
    final controlador = _controlador;

    if (controlador != null && controlador.value.isStreamingImages) {
      await controlador.stopImageStream();
    }

    await _servidor.detener();
    _conversor.detener();
    await ConversorNativo.mantenerPantallaEncendida(false);

    if (!mounted) return;

    setState(() {
      _transmitiendo = false;
      _estado = 'Camara lista';
      _fps = 0;
    });
  }

  @override
  void dispose() {
    _servidor.detener();
    _conversor.detener();
    ConversorNativo.mantenerPantallaEncendida(false);
    _controlador?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controlador = _controlador;
    final camaraLista = controlador != null && controlador.value.isInitialized;

    final urlStream = _direccionIp != null
        ? 'http://$_direccionIp:$puertoServidor/video'
        : null;

    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            _encabezado(context),
            Expanded(child: _visor(context, controlador, camaraLista)),
            _panel(context, urlStream, camaraLista),
          ],
        ),
      ),
    );
  }

  Widget _encabezado(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(
        Espacio.lg,
        Espacio.md,
        Espacio.lg,
        Espacio.md,
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text('Camcel', style: Theme.of(context).textTheme.titleMedium),
          TextButton.icon(
            onPressed: _abrirPantallaQr,
            icon: const Icon(Icons.qr_code_2, size: 20),
            label: const Text('Conectar por QR'),
            style: TextButton.styleFrom(
              foregroundColor: Paleta.tenue,
              shape: const RoundedRectangleBorder(),
              textStyle: Theme.of(context).textTheme.labelSmall,
            ),
          ),
        ],
      ),
    );
  }

  Widget _visor(
    BuildContext context,
    CameraController? controlador,
    bool camaraLista,
  ) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: Espacio.lg),
      decoration: BoxDecoration(
        color: Colors.black,
        border: Border.all(color: Paleta.filete),
      ),
      clipBehavior: Clip.hardEdge,
      child: camaraLista
          ? CameraPreview(controlador!)
          : Center(
              child: Text(
                _estado,
                textAlign: TextAlign.center,
                style: const TextStyle(color: Paleta.tenue),
              ),
            ),
    );
  }

  Widget _panel(BuildContext context, String? urlStream, bool camaraLista) {
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(top: Espacio.lg),
      decoration: const BoxDecoration(
        color: Paleta.superficie,
        border: Border(top: BorderSide(color: Paleta.filete)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(
              Espacio.lg,
              Espacio.lg,
              Espacio.lg,
              Espacio.md,
            ),
            child: Row(
              children: [
                Expanded(child: _botonPrimario(context, camaraLista)),
                const SizedBox(width: Espacio.md),
                Expanded(child: _botonSecundario(context, camaraLista)),
              ],
            ),
          ),
          _campoEstado(context),
          Padding(
            padding: const EdgeInsets.fromLTRB(
              Espacio.lg,
              Espacio.md,
              Espacio.lg,
              Espacio.lg,
            ),
            child: Column(
              children: [
                Fila(
                  'Direccion',
                  control: urlStream != null
                      ? SelectableText(
                          urlStream,
                          style: Theme.of(context).textTheme.bodyMedium,
                        )
                      : const Text(
                          'Buscando IP',
                          style: TextStyle(color: Paleta.filete),
                        ),
                ),
                Fila(
                  'Resolucion',
                  control: DropdownButtonHideUnderline(
                    child: DropdownButton<String>(
                      value: _resolucionSeleccionada,
                      dropdownColor: Paleta.campo,
                      borderRadius: BorderRadius.zero,
                      isDense: true,
                      icon: const Icon(Icons.expand_more, color: Paleta.tenue),
                      style: Theme.of(context).textTheme.bodyMedium,
                      items: resolucionesDisponibles.keys
                          .map(
                            (nombre) => DropdownMenuItem(
                              value: nombre,
                              child: Text(nombre),
                            ),
                          )
                          .toList(),
                      onChanged: (valor) {
                        if (valor != null) _cambiarResolucion(valor);
                      },
                    ),
                  ),
                ),
                Fila('Compresion', valor: _transmitiendo ? _textoSalida() : null),
                Fila(
                  'Fps',
                  valor: _transmitiendo ? _fps.toStringAsFixed(0) : null,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  /// Mismo campo de color que el veredicto de la app de escritorio: se
  /// lee de reojo, a varios metros, sin tener que enfocar la vista.
  Widget _campoEstado(BuildContext context) {
    final fondo = _transmitiendo ? Paleta.correcto : Paleta.filete;
    final tinta = _transmitiendo ? Paleta.sobreCampo : Paleta.tenue;

    return Container(
      width: double.infinity,
      color: fondo,
      padding: const EdgeInsets.symmetric(
        horizontal: Espacio.lg,
        vertical: Espacio.lg,
      ),
      child: Text(
        _transmitiendo ? 'En el aire' : _estado,
        style: Theme.of(context).textTheme.displaySmall?.copyWith(color: tinta),
      ),
    );
  }

  String _textoSalida() {
    return _usandoConversorNativo ? 'Nativa' : 'Dart';
  }

  Widget _botonPrimario(BuildContext context, bool camaraLista) {
    final habilitado = camaraLista && !_transmitiendo;

    return SizedBox(
      height: 46,
      child: FilledButton(
        onPressed: habilitado ? _alternarTransmision : null,
        style: FilledButton.styleFrom(
          backgroundColor: Paleta.texto,
          foregroundColor: Paleta.superficie,
          disabledBackgroundColor: Paleta.texto,
          disabledForegroundColor: Paleta.filete,
          shape: const RoundedRectangleBorder(),
          textStyle: Theme.of(context).textTheme.labelLarge,
        ),
        child: const Text('Empezar'),
      ),
    );
  }

  Widget _botonSecundario(BuildContext context, bool camaraLista) {
    final habilitado = camaraLista && _transmitiendo;

    return SizedBox(
      height: 46,
      child: OutlinedButton(
        onPressed: habilitado ? _alternarTransmision : null,
        style: OutlinedButton.styleFrom(
          foregroundColor: Paleta.texto,
          disabledForegroundColor: Paleta.filete,
          side: const BorderSide(color: Paleta.filete),
          shape: const RoundedRectangleBorder(),
          textStyle: Theme.of(context).textTheme.labelLarge,
        ),
        child: const Text('Parar'),
      ),
    );
  }
}
