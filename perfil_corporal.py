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

# Campo de vision horizontal tipico de una camara trasera de celular.
# Es una suposicion: varia entre modelos y con el zoom. De lo acertada
# que sea depende la correccion de perspectiva, asi que conviene poder
# cambiarla desde fuera.
FOV_TIPICO = 67.0

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


def focal_en_px(ancho_imagen, fov_grados=None):
    """Distancia focal equivalente, en pixeles.

    Es lo que convierte angulos en pixeles: con ella se puede pasar de
    la imagen al espacio y al reves. Se deduce del campo de vision,
    que es un dato aproximado de la camara, no una medida exacta.
    """

    fov = FOV_TIPICO if fov_grados is None else fov_grados

    return (ancho_imagen / 2) / math.tan(math.radians(fov) / 2)


def _normalizar(vector):
    largo = math.sqrt(sum(c * c for c in vector))

    if largo < 1e-9:
        return (0.0, 0.0, 1.0)

    return tuple(c / largo for c in vector)


def _cortar_rayo_con_esfera(rayo, centro, radio):
    """Los puntos del rayo que quedan a `radio` del centro.

    El rayo sale de la camara (el origen), asi que el punto buscado es
    s*rayo. Imponer que diste `radio` del otro extremo del hueso da
    una ecuacion de segundo grado en s.

    Devuelve las DOS soluciones porque las dos son fisicamente validas:
    una imagen no distingue si la articulacion se fue hacia la camara o
    hacia el fondo. Quien llama decide.
    """

    proyeccion = sum(r * c for r, c in zip(rayo, centro))
    distancia_centro2 = sum(c * c for c in centro)

    discriminante = proyeccion ** 2 - distancia_centro2 + radio ** 2

    if discriminante < 0:
        # El hueso no alcanza ese rayo: la escala o la deteccion no
        # cuadran. Mejor no devolver un numero inventado.
        return []

    raiz = math.sqrt(discriminante)

    return [
        tuple(s * c for c in rayo)
        for s in (proyeccion - raiz, proyeccion + raiz)
        if s > 0
    ]


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
    # Reconstruccion 3D con perspectiva
    # ---------------------------------

    def distancia_camara_cm(self, puntos, ancho_imagen, fov_grados=FOV_TIPICO):
        """A que distancia esta la persona, en centimetros.

        Se despeja del ancho de caderas: se sabe cuanto mide en
        centimetros (del perfil) y cuanto ocupa en pixeles ahora. Cuanto
        mas pequeno se ve, mas lejos esta.

        Arrastra el error de la estatura extrapolada, asi que sirve
        para informar, no para reconstruir: la reconstruccion usa
        anchos de cadera, donde ese error se cancela.
        """

        en_caderas = self.distancia_camara_en_caderas(
            puntos, ancho_imagen, fov_grados
        )
        caderas_cm = self.en_cm("ancho_caderas")

        if en_caderas is None or caderas_cm < 1e-6:
            return None

        return en_caderas * caderas_cm

    def distancia_camara_en_caderas(self, puntos, ancho_imagen,
                                    fov_grados=FOV_TIPICO):
        """Distancia a la camara medida en anchos de cadera.

        Esta es la unidad que usa la reconstruccion. Tomar el ancho de
        cadera como patron, en vez del centimetro, quita de en medio la
        estatura: su error de extrapolacion afectaria por igual al
        hueso y a la distancia, y al dividirse se cancela. Sin esto, un
        hueso supuesto un 5% mas largo de lo real obliga a inclinarlo
        para que cuadre, e inventa grados que no existen.
        """

        caderas_px = distancia(puntos[CADERA_I], puntos[CADERA_D])

        if caderas_px < 1e-6:
            return None

        return focal_en_px(ancho_imagen, fov_grados) / caderas_px

    def reconstruir_segmento(self, puntos, nombre, ancho_imagen, alto_imagen,
                             fov_grados=FOV_TIPICO):
        """Devuelve los dos extremos del segmento en 3D, en centimetros.

        El truco: del extremo cercano se sabe la profundidad (la da el
        ancho de caderas). Del lejano solo se sabe la RECTA por la que
        se ve, porque una imagen no da profundidad. Pero tambien se
        sabe cuanto mide el hueso, y sobre esa recta solo hay dos
        puntos a esa distancia exacta del primero. Es cortar una recta
        con una esfera: se resuelve con una ecuacion de segundo grado,
        sin aproximar nada.

        Las coordenadas van en anchos de cadera, no en centimetros,
        para que el error de la estatura no entre aqui. Para pasarlas a
        centimetros, multiplicar por en_cm("ancho_caderas").

        Las dos soluciones son fisicamente validas: una imagen no
        distingue si la rodilla se adelanto o se fue hacia atras. Se
        elige la de MENOR inclinacion, que es la lectura prudente: de
        dos explicaciones compatibles con lo que se ve, la menos
        extrema. Quedarse siempre con la cercana inventaba 26 grados
        de flexion en una pierna completamente recta.
        """

        a, b, referencia = self._extremos(puntos, nombre)

        caderas_ref = self.segmentos_px.get("ancho_caderas", 0.0)

        if caderas_ref < 1e-6:
            return None

        # Longitud del hueso expresada en anchos de cadera.
        largo = self.segmentos_px.get(referencia, 0.0) / caderas_ref

        profundidad = self.distancia_camara_en_caderas(
            puntos, ancho_imagen, fov_grados
        )

        if largo < 1e-6 or not profundidad:
            return None

        foco = focal_en_px(ancho_imagen, fov_grados)
        centro = (ancho_imagen / 2, alto_imagen / 2)

        # Extremo cercano: se conoce su profundidad, asi que basta
        # deshacer la proyeccion.
        proximal = (
            (a[0] - centro[0]) * profundidad / foco,
            (a[1] - centro[1]) * profundidad / foco,
            profundidad,
        )

        # Extremo lejano: solo se conoce su direccion.
        rayo = _normalizar((b[0] - centro[0], b[1] - centro[1], foco))

        candidatos = _cortar_rayo_con_esfera(rayo, proximal, largo)

        if not candidatos:
            return None

        def inclinacion(punto):
            vector = tuple(p - q for p, q in zip(punto, proximal))
            modulo = math.sqrt(sum(c * c for c in vector))

            if modulo < 1e-9:
                return math.inf

            return math.acos(min(1.0, abs(vector[1]) / modulo))

        return proximal, min(candidatos, key=inclinacion)

    def candidatos_segmento(self, puntos, nombre, ancho_imagen, alto_imagen,
                            fov_grados=FOV_TIPICO):
        """Las dos reconstrucciones posibles del segmento.

        Se exponen las dos porque elegir entre ellas no es un problema
        de geometria sino de contexto: ver SeguidorSegmentos.
        """

        a, b, referencia = self._extremos(puntos, nombre)

        caderas_ref = self.segmentos_px.get("ancho_caderas", 0.0)

        if caderas_ref < 1e-6:
            return []

        largo = self.segmentos_px.get(referencia, 0.0) / caderas_ref

        profundidad = self.distancia_camara_en_caderas(
            puntos, ancho_imagen, fov_grados
        )

        if largo < 1e-6 or not profundidad:
            return []

        foco = focal_en_px(ancho_imagen, fov_grados)
        centro = (ancho_imagen / 2, alto_imagen / 2)

        proximal = (
            (a[0] - centro[0]) * profundidad / foco,
            (a[1] - centro[1]) * profundidad / foco,
            profundidad,
        )

        rayo = _normalizar((b[0] - centro[0], b[1] - centro[1], foco))

        return [
            (proximal, distal)
            for distal in _cortar_rayo_con_esfera(rayo, proximal, largo)
        ]

    def angulo_con_vertical_3d(self, puntos, nombre, ancho_imagen, alto_imagen,
                               fov_grados=FOV_TIPICO):
        """Inclinacion real del segmento respecto a la vertical.

        A diferencia de angulo_con_vertical, esta si corrige la
        perspectiva, y por eso sirve tambien para las piernas.
        """

        extremos = self.reconstruir_segmento(
            puntos, nombre, ancho_imagen, alto_imagen, fov_grados
        )

        if extremos is None:
            return None

        (ax, ay, az), (bx, by, bz) = extremos

        vector = (bx - ax, by - ay, bz - az)
        largo = math.sqrt(sum(c * c for c in vector))

        if largo < 1e-6:
            return None

        # El eje vertical de la imagen es y, que crece hacia abajo.
        return math.degrees(math.acos(min(1.0, abs(vector[1]) / largo)))

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


def inclinacion_de(extremos):
    """Angulo del segmento con la vertical, en grados."""

    (ax, ay, az), (bx, by, bz) = extremos

    vector = (bx - ax, by - ay, bz - az)
    modulo = math.sqrt(sum(c * c for c in vector))

    if modulo < 1e-9:
        return None

    return math.degrees(math.acos(min(1.0, abs(vector[1]) / modulo)))


class SeguidorSegmentos:
    """Resuelve la ambiguedad de profundidad siguiendo el movimiento.

    Reconstruir un hueso desde una sola camara deja siempre dos
    respuestas: la articulacion pudo irse hacia la camara o hacia el
    fondo, y en la imagen se ven igual. Ninguna regla fija acierta
    siempre; probado contra verdad conocida, quedarse con la cercana
    inventa 26 grados en una pierna recta, y quedarse con la menos
    inclinada se equivoca en 26 cuando la flexion es real.

    El seguimiento temporal TAMPOCO lo resuelve, y conviene saberlo
    antes de confiar en el. Medido contra verdad conocida sobre una
    sentadilla completa, fotograma a fotograma y a 2 grados por frame:

        flexion real     estimado
           0 grados         0.0     <- exacto
          10 grados        10.0     <- exacto
          20 grados         6.6
          30 grados         4.2
          60 grados        39.4

    La razon es geometrica, no un fallo de implementacion: las dos
    reconstrucciones parten del mismo punto con la pierna recta y se
    separan de forma continua segun se flexiona. Las dos son curvas
    continuas, asi que "seguir la que estaba cerca" no distingue la
    buena. Es la ambiguedad del cubo de Necker: de frente, una pierna
    que se adelanta y otra que se atrasa producen la misma imagen.

    Lo que si puede romper el empate son restricciones anatomicas: la
    rodilla solo dobla hacia un lado, y en sentadilla el pie no se
    despega del suelo. Reconstruyendo femur y tibia a la vez, de las
    cuatro combinaciones solo una cumple ambas. Eso esta sin hacer.

    Mientras tanto: fiable con el tronco y con flexiones pequenas,
    no fiable con piernas muy flexionadas.
    """

    def __init__(self, perfil, ancho_imagen, alto_imagen,
                 fov_grados=FOV_TIPICO):
        self.perfil = perfil
        self.ancho_imagen = ancho_imagen
        self.alto_imagen = alto_imagen
        self.fov_grados = fov_grados

        self._ultima_z = {}

    def reiniciar(self):
        """Olvida el seguimiento. Usar si se perdio a la persona: con
        un hueco en el video, la continuidad ya no vale como pista."""
        self._ultima_z.clear()

    def seguir(self, puntos, nombre):
        """Reconstruye el segmento continuando el movimiento anterior."""

        candidatos = self.perfil.candidatos_segmento(
            puntos, nombre, self.ancho_imagen, self.alto_imagen,
            self.fov_grados,
        )

        if not candidatos:
            return None

        previa = self._ultima_z.get(nombre)

        if previa is None:
            # Arranque: sin historia, se toma la lectura prudente. Es
            # correcta mientras se empiece de pie.
            # El None se filtra aparte y no con `or`: una inclinacion
            # de 0.0 grados es falsa para Python, y con `or` se
            # descartaba justo la lectura perfecta de la pose erguida.
            def prudencia(extremos):
                angulo = inclinacion_de(extremos)

                return math.inf if angulo is None else angulo

            elegido = min(candidatos, key=prudencia)
        else:
            # La profundidad del extremo movil no puede dar saltos.
            elegido = min(
                candidatos, key=lambda e: abs(e[1][2] - previa)
            )

        self._ultima_z[nombre] = elegido[1][2]

        return elegido

    def angulo_con_vertical(self, puntos, nombre):
        extremos = self.seguir(puntos, nombre)

        return None if extremos is None else inclinacion_de(extremos)


class ReconstructorPierna:
    """Reconstruye cadera-rodilla-tobillo resolviendo la ambiguedad con
    el pie como ancla.

    La ambiguedad de profundidad no se puede romper mirando un hueso
    aislado, ni siguiendolo en el tiempo: las dos reconstrucciones son
    continuas y salen del mismo sitio. Pero la pierna no es un hueso
    suelto, es una cadena, y el pie no se despega del suelo.

    Eso da la restriccion que faltaba. De las dos rodillas posibles,
    cada una obliga a un tobillo distinto; solo una deja el tobillo a
    la profundidad donde realmente esta el pie, que se mide una vez de
    pie y no cambia mientras no se den pasos.

    NO USAR TODAVIA CON VIDEO REAL. La idea funciona y la geometria es
    exacta, pero el metodo es demasiado sensible al ruido para servir:

        ruido en landmarks    error medio del angulo
             0 px                  0.0 grados
           0.5 px                 30.2 grados
             1 px                 32.0 grados
             2 px                 38.8 grados

    Con datos perfectos clava el angulo de rodilla de 180 a 80 grados
    con error 0.00. Con medio pixel de ruido deja de servir.

    El motivo es que el problema esta mal condicionado justo donde
    empieza el movimiento. Con la pierna recta, las dos soluciones
    casi coinciden, y elegir entre ellas depende de diferencias mas
    pequenas que el temblor de MediaPipe. Ademas el fallo no es
    gradual: equivocar la rama no da un error pequeno, da el doble de
    la inclinacion, unos 70 grados. Y como toda sentadilla empieza de
    pie, se entra siempre por el peor caso.

    Se probo pesar la continuidad entre fotogramas para castigar los
    saltos de rama. No basta: con movimiento fluido a 30 fps y medio
    pixel de ruido el error medio sigue en 30 grados, y encima empeora
    el caso limpio. Se quito: no se deja un ajuste que perjudica.

    Equivocar el campo de vision, en cambio, apenas importa: ±10
    grados de FOV cuestan unos 3 de angulo. El eslabon debil es el
    ruido, no la suposicion de la camara.

    Para que esto fuera utilizable harian falta mas restricciones que
    un solo fotograma no da: una segunda camara, o un modelo del
    cuerpo entero que resuelva todas las articulaciones a la vez en
    lugar de hueso por hueso.
    """

    def __init__(self, perfil, ancho_imagen, alto_imagen,
                 fov_grados=FOV_TIPICO):
        self.perfil = perfil
        self.ancho_imagen = ancho_imagen
        self.alto_imagen = alto_imagen
        self.fov_grados = fov_grados

        self._z_tobillo = {}

    def anclar(self, puntos, lado):
        """Fija donde esta el pie. Llamar con la persona de pie."""

        cadena = self._reconstruir(puntos, lado, z_ancla=None)

        if cadena is None:
            return False

        self._z_tobillo[lado] = cadena[2][2]

        return True

    def anclado(self, lado):
        return lado in self._z_tobillo

    def soltar(self):
        """Olvida el ancla: usar si la persona dio un paso."""
        self._z_tobillo.clear()

    def _rayo_a(self, punto):
        foco = focal_en_px(self.ancho_imagen, self.fov_grados)

        return _normalizar((
            punto[0] - self.ancho_imagen / 2,
            punto[1] - self.alto_imagen / 2,
            foco,
        ))

    def _reconstruir(self, puntos, lado, z_ancla):
        """Mejor cadena cadera-rodilla-tobillo, en anchos de cadera."""

        caderas_ref = self.perfil.segmentos_px.get("ancho_caderas", 0.0)

        if caderas_ref < 1e-6:
            return None

        largo_tibia = self.perfil.segmentos_px.get("tibia", 0.0) / caderas_ref

        if largo_tibia < 1e-6:
            return None

        candidatos_femur = self.perfil.candidatos_segmento(
            puntos, f"femur_{lado}", self.ancho_imagen, self.alto_imagen,
            self.fov_grados,
        )

        if not candidatos_femur:
            return None

        indice_tobillo = TOBILLO_I if lado == "i" else TOBILLO_D
        rayo_tobillo = self._rayo_a(puntos[indice_tobillo])

        mejor = None
        mejor_coste = math.inf

        for cadera, rodilla in candidatos_femur:
            tobillos = _cortar_rayo_con_esfera(
                rayo_tobillo, rodilla, largo_tibia
            )

            for tobillo in tobillos:
                if z_ancla is None:
                    # Sin ancla todavia: se prefiere la cadena mas
                    # estirada, que es la correcta estando de pie.
                    coste = -abs(tobillo[1] - cadera[1])
                else:
                    coste = abs(tobillo[2] - z_ancla)

                if coste < mejor_coste:
                    mejor_coste = coste
                    mejor = (cadera, rodilla, tobillo)

        return mejor

    def pierna(self, puntos, lado):
        """Cadera, rodilla y tobillo en 3D. None si no se puede."""

        return self._reconstruir(puntos, lado, self._z_tobillo.get(lado))

    def angulo_rodilla(self, puntos, lado):
        """Angulo real de la rodilla, en grados. 180 = pierna recta."""

        cadena = self.pierna(puntos, lado)

        if cadena is None:
            return None

        cadera, rodilla, tobillo = cadena

        muslo = tuple(c - r for c, r in zip(cadera, rodilla))
        pierna = tuple(t - r for t, r in zip(tobillo, rodilla))

        return _angulo_entre(muslo, pierna)

    def inclinacion_femur(self, puntos, lado):
        cadena = self.pierna(puntos, lado)

        if cadena is None:
            return None

        return inclinacion_de((cadena[0], cadena[1]))


def _angulo_entre(u, v):
    modulo_u = math.sqrt(sum(c * c for c in u))
    modulo_v = math.sqrt(sum(c * c for c in v))

    if modulo_u < 1e-9 or modulo_v < 1e-9:
        return None

    coseno = sum(a * b for a, b in zip(u, v)) / (modulo_u * modulo_v)

    return math.degrees(math.acos(min(1.0, max(-1.0, coseno))))
