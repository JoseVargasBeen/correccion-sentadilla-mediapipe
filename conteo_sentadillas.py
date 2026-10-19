"""
conteo_sentadillas.py

Cuenta repeticiones y juzga cada una, a partir del angulo de rodilla.

Una sentadilla es un ciclo: de pie, bajar, llegar abajo, subir, volver
a estar de pie. Contar repeticiones es reconocer ese ciclo; juzgarlas
es mirar que paso dentro.

Por que una maquina de estados y no un contador de minimos
-----------------------------------------------------------
La tentacion es contar cada vez que el angulo baja de un umbral. Eso
cuenta de mas: el angulo tiembla unos grados por el ruido de los
landmarks, y cruzando el umbral de ida y vuelta suma varias
repeticiones por cada una real. Con estados hay que recorrer el ciclo
entero para sumar: bajar no cuenta si no se vuelve a subir.

Los umbrales llevan margen por la misma razon que la alerta de
postura: entrar cuesta mas que salir, asi un temblor en el limite no
hace parpadear el estado.
"""

from dataclasses import dataclass

DE_PIE = "de pie"
BAJANDO = "bajando"
ABAJO = "abajo"
SUBIENDO = "subiendo"

ESTADOS = (DE_PIE, BAJANDO, ABAJO, SUBIENDO)

# Resultado de una transicion que no es un estado: cerrar el ciclo.
CERRAR = object()

# Rodilla estirada. No se usa 180 porque nadie se para con la pierna
# perfectamente recta, y los landmarks rara vez lo reportan.
ANGULO_DE_PIE = 160.0

# Muslo paralelo al suelo, el criterio habitual de "sentadilla
# completa". Es un punto de partida: lo correcto es medirlo con la
# persona, como se hizo con el encorvamiento.
ANGULO_PROFUNDIDAD = 100.0

# Por debajo de esto se considera que hubo intento de sentadilla,
# aunque no llegue a la profundidad buena. Sirve para separar una
# repeticion floja, que hay que contar y avisar, de agacharse un poco
# a recoger algo, que no es una repeticion de nada.
ANGULO_INTENTO = 140.0

# Cuanto hay que alejarse del umbral para cambiar de estado.
MARGEN = 8.0


@dataclass
class Repeticion:
    """Resumen de una repeticion terminada."""

    numero: int
    angulo_minimo: float
    profunda: bool
    hubo_alerta_postura: bool

    def resumen(self):
        if not self.profunda:
            return "Te faltó bajar"

        if self.hubo_alerta_postura:
            return "Buena profundidad, cuida la espalda"

        return "Bien"


class ContadorSentadillas:
    """Sigue el ciclo de la sentadilla y cuenta las repeticiones.

    Se alimenta fotograma a fotograma con el angulo de rodilla. Cuando
    se completa una repeticion devuelve su resumen; el resto del
    tiempo devuelve None.

    >>> c = ContadorSentadillas()
    >>> c.estado
    'de pie'

    Bajar sola no cuenta: hay que volver a subir.

    >>> c.actualizar(90.0) is None
    True
    >>> c.estado
    'abajo'
    >>> c.repeticiones
    0

    Al volver arriba se cierra el ciclo.

    >>> rep = c.actualizar(170.0)
    >>> rep.numero, rep.profunda
    (1, True)
    >>> c.repeticiones
    1

    Una bajada corta cuenta como repeticion, pero marcada.

    >>> c.actualizar(130.0) is None
    True
    >>> rep = c.actualizar(170.0)
    >>> rep.profunda, rep.resumen()
    (False, 'Te faltó bajar')
    """

    def __init__(self, angulo_de_pie=ANGULO_DE_PIE,
                 angulo_profundidad=ANGULO_PROFUNDIDAD,
                 angulo_intento=ANGULO_INTENTO, margen=MARGEN):
        self.angulo_de_pie = angulo_de_pie
        self.angulo_profundidad = angulo_profundidad
        self.angulo_intento = angulo_intento
        self.margen = margen

        self.estado = DE_PIE
        self.repeticiones = 0
        self.ultima = None

        self._angulo_minimo = None
        self._alerta_en_curso = False

    def reiniciar(self):
        self.__init__(self.angulo_de_pie, self.angulo_profundidad,
                      self.angulo_intento, self.margen)

    def actualizar(self, angulo_rodilla, alerta_postura=False):
        """Procesa un fotograma. Devuelve la Repeticion si se cerro."""

        if angulo_rodilla is None:
            return None

        # Lo que pase durante el movimiento se recuerda hasta el final
        # de la repeticion: una espalda mal puesta a mitad de bajada
        # cuenta aunque al llegar arriba ya este corregida.
        if self.estado != DE_PIE:
            self._alerta_en_curso = self._alerta_en_curso or alerta_postura

            if self._angulo_minimo is None:
                self._angulo_minimo = angulo_rodilla
            else:
                self._angulo_minimo = min(self._angulo_minimo, angulo_rodilla)

        # Se encadenan transiciones hasta que el estado se estabilice.
        # Con video continuo casi siempre es una sola, pero si se cae
        # un fotograma el angulo puede saltar medio recorrido, y
        # avanzar un unico paso dejaria la maquina atrasada para
        # siempre. El tope corta cualquier ciclo inesperado.
        for _ in range(len(ESTADOS) + 1):
            siguiente = self._transicion(angulo_rodilla, alerta_postura)

            if siguiente is CERRAR:
                return self._cerrar()

            if siguiente is None:
                return None

            self.estado = siguiente

        return None

    def _transicion(self, angulo_rodilla, alerta_postura):
        """Siguiente estado, o None si hay que quedarse donde esta."""

        if self.estado == DE_PIE:
            if angulo_rodilla < self.angulo_de_pie - self.margen:
                self._angulo_minimo = angulo_rodilla
                self._alerta_en_curso = alerta_postura
                return BAJANDO

        elif self.estado == BAJANDO:
            if angulo_rodilla < self.angulo_profundidad:
                return ABAJO

            if angulo_rodilla > self.angulo_de_pie:
                # Volvio arriba sin llegar al fondo. Si bajo lo
                # suficiente cuenta como repeticion floja; si apenas
                # se movio, no era una sentadilla.
                return CERRAR

        elif self.estado == ABAJO:
            if angulo_rodilla > self.angulo_profundidad + self.margen:
                return SUBIENDO

        elif self.estado == SUBIENDO:
            if angulo_rodilla > self.angulo_de_pie:
                return CERRAR

        return None

    def _cerrar(self):
        minimo = self._angulo_minimo

        self.estado = DE_PIE
        self._angulo_minimo = None
        alerta = self._alerta_en_curso
        self._alerta_en_curso = False

        # Un movimiento que ni se acerco a una sentadilla no se cuenta.
        if minimo is None or minimo > self.angulo_intento:
            return None

        self.repeticiones += 1

        self.ultima = Repeticion(
            numero=self.repeticiones,
            angulo_minimo=minimo,
            profunda=minimo <= self.angulo_profundidad,
            hubo_alerta_postura=alerta,
        )

        return self.ultima
