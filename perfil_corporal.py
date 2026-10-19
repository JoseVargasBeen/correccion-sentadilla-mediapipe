"""
perfil_corporal.py

Mide una vez las proporciones de la persona y las guarda, para que el
analisis de postura tenga una referencia propia contra la que comparar.

Por que hace falta
------------------
De perfil, el encorvamiento se ve directo: la cabeza se adelanta
respecto al tronco. De frente no, porque ese adelantamiento ocurre
hacia la camara y no hacia los lados.

Pero deja una huella medible: al encorvarse, el tronco se inclina
fuera del plano de la camara y su imagen se ACORTA. Es el mismo efecto
por el que un lapiz apuntando hacia uno se ve mas corto. Si se sabe
cuanto media ese tronco con la persona derecha, un acortamiento
delata la joroba sin necesidad de verla de lado.

La trampa de los pixeles
------------------------
Un tronco tambien se acorta en pixeles si la persona se aleja de la
camara. Por eso aqui nada se compara en pixeles crudos: se compara la
RAZON entre el tronco y el ancho de caderas. Al alejarse, ambos
encogen igual y la razon no cambia; al encorvarse, solo encoge el
tronco. Se usan las caderas y no los hombros porque al encorvarse los
hombros se redondean hacia adelante y tambien se acortan, lo que
borraria parte de la senal.

La estatura
-----------
Sirve para poner las medidas en centimetros, util para informes y para
detectar disparates. El analisis de postura NO depende de ella: todo
lo que decide funciona con razones adimensionales.
"""

import json
import math
import os
from dataclasses import asdict, dataclass, field

# Indices de MediaPipe Pose (33 puntos).
NARIZ = 0
HOMBRO_I, HOMBRO_D = 11, 12
CODO_I, CODO_D = 13, 14
MUNECA_I, MUNECA_D = 15, 16
CADERA_I, CADERA_D = 23, 24
RODILLA_I, RODILLA_D = 25, 26
TOBILLO_I, TOBILLO_D = 27, 28

ARCHIVO_PERFIL = "perfil_corporal.json"

# Fraccion de la estatura a la que queda la nariz por encima del
# tobillo, en proporciones antropometricas medias. Solo se usa para
# pasar a centimetros; ninguna decision de postura depende de este
# numero.
FRACCION_NARIZ_TOBILLO = 0.89

# Cuanto puede acortarse la razon del tronco antes de considerarlo
# escorzo y no ruido de deteccion.
#
# Medido sobre un cuerpo sintetico de verdad conocida, inclinando el
# tronco hacia una camara a 3 m y altura de pecho:
#
#   inclinacion    razon
#      10 grados   0.998
#      15 grados   0.985
#      20 grados   0.964
#      30 grados   0.896
#      40 grados   0.794
#
# La senal es mas debil de lo que sugiere la geometria pura: al
# inclinarse, el tronco se acorta por escorzo pero a la vez se ACERCA
# a la camara, y esa magnificacion cancela cerca de la mitad del
# efecto. Por eso esto sirve para delatar inclinaciones marcadas, no
# encorvamientos sutiles: para eso sigue haciendo falta el perfil.
CAIDA_TRONCO_SOSPECHOSA = 0.93

# Margenes para dar por mala una deteccion: un segmento que mide la
# mitad o el doble que en la referencia no es una postura, es un
# landmark perdido.
FACTOR_MIN_SANO = 0.55
FACTOR_MAX_SANO = 1.8


def distancia(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def punto_medio(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def medir_segmentos(puntos):
    """Longitudes en pixeles de los segmentos del cuerpo.

    Los pares se promedian entre lado izquierdo y derecho: un brazo
    puede quedar peor detectado que el otro, y el promedio absorbe
    parte de ese ruido.
    """

    hombro = punto_medio(puntos[HOMBRO_I], puntos[HOMBRO_D])
    cadera = punto_medio(puntos[CADERA_I], puntos[CADERA_D])

    def promedio(par_i, par_j):
        return (
            distancia(puntos[par_i[0]], puntos[par_i[1]])
            + distancia(puntos[par_j[0]], puntos[par_j[1]])
        ) / 2

    return {
        "ancho_hombros": distancia(puntos[HOMBRO_I], puntos[HOMBRO_D]),
        "ancho_caderas": distancia(puntos[CADERA_I], puntos[CADERA_D]),
        "tronco": distancia(hombro, cadera),
        "cuello": distancia(puntos[NARIZ], hombro),
        "brazo": promedio((HOMBRO_I, CODO_I), (HOMBRO_D, CODO_D)),
        "antebrazo": promedio((CODO_I, MUNECA_I), (CODO_D, MUNECA_D)),
        "femur": promedio((CADERA_I, RODILLA_I), (CADERA_D, RODILLA_D)),
        "tibia": promedio((RODILLA_I, TOBILLO_I), (RODILLA_D, TOBILLO_D)),
    }


def razon_tronco(segmentos):
    """Tronco medido en anchos de cadera. Invariante a la distancia."""

    caderas = segmentos["ancho_caderas"]

    if caderas < 1e-6:
        return None

    return segmentos["tronco"] / caderas


@dataclass
class PerfilCorporal:
    """Proporciones de una persona, medidas en la pose de referencia."""

    estatura_cm: float
    segmentos_px: dict
    razon_tronco: float
    cm_por_px: float
    muestras: int = 0
    notas: dict = field(default_factory=dict)

    # ---------------------------------
    # Construccion
    # ---------------------------------

    @classmethod
    def desde_muestras(cls, estatura_cm, lista_puntos):
        """Construye el perfil promediando varias capturas.

        Se promedia porque los landmarks bailan unos pixeles entre
        frames; con una sola captura el perfil heredaria ese temblor
        como si fuera la medida buena.
        """

        if not lista_puntos:
            raise ValueError("No hay muestras para construir el perfil")

        acumulado = {}

        for puntos in lista_puntos:
            for nombre, valor in medir_segmentos(puntos).items():
                acumulado.setdefault(nombre, []).append(valor)

        segmentos = {
            nombre: sum(valores) / len(valores)
            for nombre, valores in acumulado.items()
        }

        altura_px = cls._altura_en_px(lista_puntos)
        cm_por_px = estatura_cm / altura_px if altura_px else 0.0

        return cls(
            estatura_cm=estatura_cm,
            segmentos_px=segmentos,
            razon_tronco=razon_tronco(segmentos) or 0.0,
            cm_por_px=cm_por_px,
            muestras=len(lista_puntos),
        )

    @staticmethod
    def _altura_en_px(lista_puntos):
        """Estatura aparente, extrapolando de la nariz al alto total.

        MediaPipe no da un punto en la coronilla: el mas alto es la
        nariz. Se divide por la fraccion antropometrica para completar
        la cabeza. Es una aproximacion, y por eso solo alimenta la
        conversion a centimetros.
        """

        alturas = []

        for puntos in lista_puntos:
            tobillo = punto_medio(puntos[TOBILLO_I], puntos[TOBILLO_D])
            visible = abs(tobillo[1] - puntos[NARIZ][1])

            if visible > 1:
                alturas.append(visible / FRACCION_NARIZ_TOBILLO)

        if not alturas:
            return 0.0

        return sum(alturas) / len(alturas)

    # ---------------------------------
    # Uso en tiempo real
    # ---------------------------------

    def en_cm(self, nombre):
        """Longitud estimada de un segmento, en centimetros."""
        return self.segmentos_px.get(nombre, 0.0) * self.cm_por_px

    def medidas_en_cm(self):
        return {n: self.en_cm(n) for n in self.segmentos_px}

    def acortamiento_tronco(self, puntos):
        """Cuanto se ha acortado el tronco respecto a la referencia.

        1.0 = igual que de pie y derecho. Por debajo de 1.0 el tronco
        se esta viendo mas corto, lo que significa que se inclino o se
        encorvo hacia la camara. Devuelve None si no se puede medir.
        """

        if not self.razon_tronco:
            return None

        actual = razon_tronco(medir_segmentos(puntos))

        if actual is None:
            return None

        return actual / self.razon_tronco

    # ---------------------------------
    # Angulos reales a partir del escorzo
    # ---------------------------------

    def escala_actual(self, puntos):
        """Cuanto mide ahora la persona respecto a la referencia.

        Se mide con el ancho de caderas, que no cambia al inclinarse:
        es lo que permite separar "se alejo" de "se inclino".
        """

        caderas_ref = self.segmentos_px.get("ancho_caderas", 0.0)

        if caderas_ref < 1e-6:
            return None

        return distancia(puntos[CADERA_I], puntos[CADERA_D]) / caderas_ref

    def _extremos(self, puntos, nombre):
        """Los dos puntos de un segmento, y su longitud de referencia."""

        if nombre == "tronco":
            return (
                punto_medio(puntos[HOMBRO_I], puntos[HOMBRO_D]),
                punto_medio(puntos[CADERA_I], puntos[CADERA_D]),
                "tronco",
            )

        pares = {
            "femur_i": (CADERA_I, RODILLA_I, "femur"),
            "femur_d": (CADERA_D, RODILLA_D, "femur"),
            "tibia_i": (RODILLA_I, TOBILLO_I, "tibia"),
            "tibia_d": (RODILLA_D, TOBILLO_D, "tibia"),
        }

        if nombre not in pares:
            raise KeyError(nombre)

        a, b, referencia = pares[nombre]

        return puntos[a], puntos[b], referencia

    def angulo_fuera_de_plano(self, puntos, nombre):
        """Cuanto se sale un segmento del plano de la camara, en grados.

        Si un hueso de 45 cm se ve midiendo 40, es que apunta hacia la
        camara (o hacia el fondo): el coseno de esa razon da el angulo.
        Asi se recupera informacion de profundidad desde una sola
        camara, sin sensor 3D.

        Ojo: devuelve la MAGNITUD, no el sentido. Inclinarse hacia la
        camara y hacia el fondo acortan igual, y una sola imagen no
        puede distinguirlos.
        """

        escala = self.escala_actual(puntos)

        if not escala:
            return None

        a, b, referencia = self._extremos(puntos, nombre)

        esperada = self.segmentos_px.get(referencia, 0.0) * escala

        if esperada < 1e-6:
            return None

        razon = distancia(a, b) / esperada

        # Por encima de 1 el segmento se ve mas largo de lo que mide:
        # es ruido de deteccion, no profundidad negativa.
        return math.degrees(math.acos(min(1.0, max(0.0, razon))))

    def angulo_con_vertical(self, puntos, nombre):
        """Inclinacion real del segmento respecto a la vertical, en 3D.

        Comparando la altura que ocupa el segmento en imagen contra su
        longitud real se obtiene la inclinacion, que de frente no se
        podria medir mirando solo el dibujo.

        FIABLE PARA EL TRONCO, NO PARA LAS PIERNAS. Medido contra un
        cuerpo sintetico de verdad conocida, con la camara a 3 m:

            inclinacion real    tronco    femur
                 10 grados        3.6      0.0
                 20 grados       15.4      0.0
                 30 grados       26.4     11.1
                 45 grados         --     31.0
                 60 grados         --     48.9

        El tronco se queda unos 4-5 grados corto; el femur se equivoca
        entre 10 y 20. La causa es la misma en los dos casos y explica
        la diferencia: al inclinarse, el segmento se acorta por escorzo
        pero tambien se ACERCA a la camara, y esa magnificacion
        disimula el acortamiento. La rodilla viaja mucho mas en
        profundidad que los hombros en proporcion a su longitud, asi
        que en la pierna la magnificacion se come casi todo el efecto.

        Corregirlo exige conocer la distancia a la camara, que con una
        sola camara sin calibrar no se tiene. Hasta entonces, usar este
        valor solo para el tronco y como estimacion por lo bajo.
        """

        escala = self.escala_actual(puntos)

        if not escala:
            return None

        a, b, referencia = self._extremos(puntos, nombre)

        esperada = self.segmentos_px.get(referencia, 0.0) * escala

        if esperada < 1e-6:
            return None

        vertical = abs(a[1] - b[1])

        return math.degrees(math.acos(min(1.0, max(0.0, vertical / esperada))))

    def hay_escorzo(self, puntos):
        """True si el tronco se ve lo bastante corto como para delatar
        una inclinacion fuera del plano de la camara."""

        factor = self.acortamiento_tronco(puntos)

        return factor is not None and factor < CAIDA_TRONCO_SOSPECHOSA

    def deteccion_coherente(self, puntos):
        """Descarta frames donde los landmarks se fueron.

        Las proporciones de una persona no cambian: si un hueso mide
        de pronto la mitad o el doble que en la referencia, lo que
        fallo es la deteccion, no el cuerpo. Sirve para no alimentar
        el analisis con basura.
        """

        # Se normaliza por el ancho de caderas para que acercarse o
        # alejarse de la camara no cuente como incoherencia.
        caderas_ref = self.segmentos_px.get("ancho_caderas", 0.0)
        caderas_ahora = distancia(puntos[CADERA_I], puntos[CADERA_D])

        if caderas_ref < 1e-6 or caderas_ahora < 1e-6:
            return False

        escala = caderas_ahora / caderas_ref

        # Cada lado se mide por separado, no promediado: con el
        # promedio, una pierna disparatada se compensaba con la buena
        # y el frame pasaba el filtro.
        pares = (
            ("femur", (CADERA_I, RODILLA_I), (CADERA_D, RODILLA_D)),
            ("tibia", (RODILLA_I, TOBILLO_I), (RODILLA_D, TOBILLO_D)),
        )

        for nombre, izquierdo, derecho in pares:
            referencia = self.segmentos_px.get(nombre, 0.0) * escala

            if referencia < 1e-6:
                continue

            for a, b in (izquierdo, derecho):
                factor = distancia(puntos[a], puntos[b]) / referencia

                if not (FACTOR_MIN_SANO <= factor <= FACTOR_MAX_SANO):
                    return False

        return True

    # ---------------------------------
    # Persistencia
    # ---------------------------------

    def guardar(self, ruta=ARCHIVO_PERFIL):
        with open(ruta, "w", encoding="utf-8") as archivo:
            json.dump(asdict(self), archivo, indent=2, ensure_ascii=False)

    @classmethod
    def cargar(cls, ruta=ARCHIVO_PERFIL):
        """Devuelve el perfil guardado, o None si no hay ninguno."""

        if not os.path.exists(ruta):
            return None

        try:
            with open(ruta, encoding="utf-8") as archivo:
                return cls(**json.load(archivo))
        except (OSError, ValueError, TypeError):
            return None
