import 'package:flutter/material.dart';

/// Misma paleta que la app de escritorio (app_pose.py), con los mismos
/// nombres por funcion y no por material: cambiar el tema no obliga a
/// renombrar nada. Las dos mitades del proyecto se ven como una sola
/// herramienta.
abstract final class Paleta {
  static const texto = Color(0xFFE6EBE6);
  static const superficie = Color(0xFF1B2623);
  static const campo = Color(0xFF24312D);
  static const tenue = Color(0xFF93A199);
  static const filete = Color(0xFF35443F);
  static const lienzo = Color(0xFF111917);
  static const correcto = Color(0xFF1F7A50);
  static const alerta = Color(0xFFC03A20);
  static const sobreCampo = Color(0xFFFFFFFF);
}

/// Escala de espaciado compartida con la version de escritorio. Usar
/// siempre estos saltos evita margenes inventados frame a frame.
abstract final class Espacio {
  static const xs = 4.0;
  static const sm = 8.0;
  static const md = 14.0;
  static const lg = 22.0;
  static const xl = 30.0;
}

const _familia = 'Barlow';

ThemeData construirTema() {
  const textoBase = TextStyle(
    fontFamily: _familia,
    color: Paleta.texto,
    fontWeight: FontWeight.w300,
  );

  return ThemeData(
    useMaterial3: true,
    brightness: Brightness.dark,
    fontFamily: _familia,
    scaffoldBackgroundColor: Paleta.lienzo,
    colorScheme: const ColorScheme.dark(
      surface: Paleta.superficie,
      onSurface: Paleta.texto,
      primary: Paleta.correcto,
      onPrimary: Paleta.sobreCampo,
      error: Paleta.alerta,
      onError: Paleta.sobreCampo,
      outline: Paleta.filete,
    ),
    textTheme: TextTheme(
      // Las cifras van en SemiBold condensado: es lo que se tiene que
      // leer a tres metros mientras se hace el ejercicio.
      displaySmall: textoBase.copyWith(
        fontSize: 34,
        fontWeight: FontWeight.w600,
        height: 1.0,
      ),
      titleMedium: textoBase.copyWith(
        fontSize: 17,
        fontWeight: FontWeight.w600,
        letterSpacing: 0.4,
      ),
      bodyMedium: textoBase.copyWith(fontSize: 15),
      labelLarge: textoBase.copyWith(
        fontSize: 15,
        fontWeight: FontWeight.w600,
        letterSpacing: 0.8,
      ),
      labelSmall: textoBase.copyWith(fontSize: 13, color: Paleta.tenue),
    ),
  );
}

/// Fila de lectura: nombre a la izquierda en tenue, valor a la derecha
/// en texto plano. Es el patron del panel de medidas de la version de
/// escritorio, y la razon de que se lea como una lista de instrumentos
/// y no como un formulario.
class Fila extends StatelessWidget {
  const Fila(this.rotulo, {super.key, this.valor, this.control});

  final String rotulo;
  final String? valor;
  final Widget? control;

  @override
  Widget build(BuildContext context) {
    final tema = Theme.of(context);

    final derecha =
        control ??
        Text(
          valor ?? '—',
          style: tema.textTheme.bodyMedium?.copyWith(
            // Un hueco sin dato no debe pesar lo mismo que una medida.
            color: valor == null ? Paleta.filete : Paleta.texto,
          ),
        );

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: Espacio.sm),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Text(
            rotulo,
            style: tema.textTheme.bodyMedium?.copyWith(color: Paleta.tenue),
          ),
          const SizedBox(width: Espacio.md),
          Expanded(
            child: Align(alignment: Alignment.centerRight, child: derecha),
          ),
        ],
      ),
    );
  }
}
