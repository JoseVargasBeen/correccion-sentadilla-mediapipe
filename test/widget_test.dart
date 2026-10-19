import 'package:flutter_test/flutter_test.dart';

import 'package:camcel/main.dart';

void main() {
  testWidgets('La app arranca sin errores', (WidgetTester tester) async {
    await tester.pumpWidget(const CamcelApp());

    // En el entorno de pruebas no hay camara real, asi que solo
    // verificamos que la pantalla principal se construya sin explotar.
    expect(find.text('Camcel'), findsOneWidget);
  });
}
