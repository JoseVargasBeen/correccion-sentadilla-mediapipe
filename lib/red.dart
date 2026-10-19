import 'dart:io';

/// Interfaces de datos moviles. Su direccion es una 10.x que el
/// operador reparte detras de su NAT: no es alcanzable desde el PC, asi
/// que nunca debe acabar en el QR.
bool _esDatosMoviles(String nombre) {
  return nombre.startsWith('rmnet') ||
      nombre.startsWith('ccmni') ||
      nombre.startsWith('pdp');
}

/// Todas las direcciones IPv4 utiles del telefono, por interfaz.
Future<Map<String, String>> direccionesPorInterfaz() async {
  final encontradas = <String, String>{};

  try {
    final interfaces = await NetworkInterface.list(
      type: InternetAddressType.IPv4,
      includeLoopback: false,
    );

    for (final interfaz in interfaces) {
      if (_esDatosMoviles(interfaz.name)) continue;

      for (final direccion in interfaz.addresses) {
        if (esRedLocal(direccion.address)) {
          encontradas[interfaz.name] = direccion.address;
          break;
        }
      }
    }
  } catch (_) {
    // Sin red utilizable se devuelve el mapa vacio.
  }

  return encontradas;
}

/// Direccion del telefono dentro de la red local (WiFi normal).
Future<String?> buscarIpLocal() async {
  final direcciones = await direccionesPorInterfaz();

  if (direcciones.isEmpty) return null;

  return direcciones.values.first;
}

/// Direccion del punto de acceso recien creado.
///
/// No se busca por prefijo: el rango depende del fabricante. En un
/// Xiaomi la red local sale en 10.65.x.x, no en la 192.168.49.1 que
/// suele citarse, y esa 10.x es indistinguible por forma de la que dan
/// los datos moviles. Lo unico fiable es cual interfaz APARECIO al
/// encender el punto de acceso, comparando contra la foto de antes.
Future<String?> buscarIpNueva(
  Map<String, String> previas, {
  Duration espera = const Duration(seconds: 8),
}) async {
  final limite = DateTime.now().add(espera);

  while (DateTime.now().isBefore(limite)) {
    await Future<void>.delayed(const Duration(milliseconds: 300));

    final actuales = await direccionesPorInterfaz();

    for (final entrada in actuales.entries) {
      final erraDistinta = previas[entrada.key] != entrada.value;

      if (erraDistinta) return entrada.value;
    }
  }

  return null;
}

bool esRedLocal(String ip) {
  return ip.startsWith('192.168.') ||
      ip.startsWith('10.') ||
      RegExp(r'^172\.(1[6-9]|2\d|3[0-1])\.').hasMatch(ip);
}
