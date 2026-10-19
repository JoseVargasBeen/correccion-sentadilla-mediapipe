"""
app_pose.py

Aplicacion de escritorio (prototipo PC) para el proyecto de
correccion de ejercicio con camara.

Reutiliza la logica de pose_angulos_sentadilla.py (deteccion de
pose + calculo de angulos y postura de espalda), pero separa el
CALCULO de la VISUALIZACION:

- El video muestra solo el skeleton (sin texto superpuesto).
- Los valores calculados (angulos, inclinacion, indice de
  encorvamiento, clasificacion de postura, FPS) se guardan como
  variables de estado y se muestran en un panel lateral aparte.

Este es un prototipo desechable para probar la logica y mostrar
avances; la version final para el usuario sera en Flutter (Fase 5).

LA CAMARA VA DE PERFIL
----------------------
Quedo decidido por medida, no por gusto. De perfil, la flexion de
rodilla es geometria plana y sale exacta, y el encorvamiento se ve
como lo que es: la cabeza adelantandose respecto al tronco.

De frente las dos cosas ocurren hacia la camara, que es la direccion
que una sola camara no mide. Se intento recuperar esa profundidad a
partir de las proporciones del cuerpo (ver perfil_corporal.py): la
geometria sale exacta sobre datos perfectos y se derrumba con medio
pixel de ruido, porque el problema esta mal condicionado justo con el
cuerpo erguido, que es como empieza todo movimiento.

Requiere: opencv-python, mediapipe, numpy, customtkinter, pillow
"""

import math
import threading
import time
import tkinter.font as tkfont

import cv2
import numpy as np
import mediapipe as mp
import customtkinter as ctk
from PIL import Image, ImageDraw, ImageTk

import conexion_qr
from conteo_sentadillas import ContadorSentadillas
from perfil_corporal import PerfilCorporal

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =================================
# Configuracion
# =================================

POSE_MODEL_PATH = "models/pose_landmarker_lite.task"

# Cuantos ms espera la interfaz entre repintados. El hilo de proceso
# va por su cuenta; esto solo marca cada cuanto se recoge lo ultimo.
MS_ENTRE_PINTADOS = 16

# Tope de ancho al mostrar el video. Ampliar mas no anade detalle y el
# coste de subir la imagen a Tk crece con el area.
ANCHO_MAX_VIDEO = 960

PREFIJO_IP_CELULAR = "192.168.1."
PUERTO_IP_WEBCAM = "8080"

# Cuanto esperar al conectar y al leer de una fuente de red antes de
# darla por perdida. Sin estos limites, OpenCV espera indefinidamente.
TIMEOUT_CONEXION_MS = 5000
TIMEOUT_LECTURA_MS = 5000

# Si se conecto pero dejan de llegar frames por este tiempo, se asume
# que la transmision murio (ej. se cerro la app del celular).
SEGUNDOS_SIN_FRAME_PARA_CORTAR = 8.0

# El giro se guarda en grados y se traduce aqui. Cada clic en las
# flechas suma o resta 90, igual que girar el celular en la mano.
ROTACIONES = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


# =================================
# Tokens de diseno
# =================================
# La aplicacion no se mira sentado frente al monitor: se mira de reojo,
# a dos o tres metros, mientras se hace el ejercicio. Todo lo de abajo
# sale de ahi.
#
# Tema oscuro. El verde de la superficie no es negro tenido: tiene
# croma propia, de la misma familia que los dos colores de estado.
#
# Contrastes comprobados sobre "superficie": texto 12.9:1, tenue 5.8:1.
# Blanco sobre los dos campos de estado: 5.3:1 y 5.4:1.

COLOR = {
    # Nombres por funcion, no por material: cambiar el tema no obliga a
    # renombrar nada.
    "texto": "#E6EBE6",
    "superficie": "#1B2623",
    "campo": "#24312D",
    "tenue": "#93A199",
    "filete": "#35443F",
    "lienzo": "#111917",
    "correcto": "#1F7A50",
    "alerta": "#C03A20",
    "sobre_campo": "#FFFFFF",
}

ESPACIO = {"xs": 4, "sm": 8, "md": 14, "lg": 22, "xl": 30}

# Los mismos colores para lo que se dibuja sobre el video. OpenCV usa
# BGR, no RGB, asi que van invertidos respecto al hex de arriba.
BGR_SKELETON = (230, 235, 230)  # #E6EBE6, el mismo texto del panel
BGR_TRONCO = (80, 122, 31)      # #1F7A50

# Bahnschrift es la version que trae Windows de la DIN: la letra de los
# planos tecnicos y la senaletica. Encaja con un instrumento de medida
# y sus digitos condensados dejan poner numeros grandes sin desbordar.
#
# Se comprueba que este instalada en vez de darlo por hecho: pedir una
# familia ausente no da error, Tk sustituye en silencio y el diseno se
# cae sin que nadie se entere.
FAMILIAS_TITULO = ("Bahnschrift SemiBold", "Bahnschrift", "Segoe UI Semibold")
FAMILIAS_TEXTO = ("Bahnschrift Light", "Bahnschrift", "Segoe UI")
FAMILIAS_CIFRA = ("Bahnschrift SemiCondensed", "Bahnschrift", "Segoe UI")


def crear_icono_giro(horario, lado=20, color=COLOR["texto"]):
    """Flecha circular dibujada a mano, no un caracter de la fuente.

    Los glifos de flecha circular (U+21BB y companhia) no estan en
    todas las tipografias: si faltan, Tk sustituye en silencio y
    aparece un cuadro vacio. Dibujarla garantiza que se vea, y permite
    que siga el mismo gris del resto del panel.
    """

    escala = 4  # se dibuja grande y se reduce: bordes limpios
    px = lado * escala

    imagen = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    dibujo = ImageDraw.Draw(imagen)

    centro = px / 2
    radio = px * 0.33
    grosor = max(1, round(px * 0.09))

    # Arco de 270 grados con el hueco arriba: asi cada punta queda a un
    # lado de la abertura y las dos flechas se distinguen de inmediato.
    # Con el hueco a un costado ambas salian casi iguales, y con el
    # hueco mas estrecho las puntas se juntaban en un borron. En PIL el
    # angulo 0 son las 3 en punto y crece en sentido horario.
    inicio, fin = 315, 225
    caja = [centro - radio, centro - radio, centro + radio, centro + radio]
    dibujo.arc(caja, start=inicio, end=fin, fill=color, width=grosor)

    # La punta va en el extremo hacia el que avanza el giro.
    angulo = math.radians(fin if horario else inicio)
    sentido = 1 if horario else -1

    px_punta = centro + radio * math.cos(angulo)
    py_punta = centro + radio * math.sin(angulo)

    # Tangente al circulo en ese punto, segun hacia donde se avanza.
    dx = -math.sin(angulo) * sentido
    dy = math.cos(angulo) * sentido

    largo = grosor * 3.2
    ancho = grosor * 2.4

    dibujo.polygon(
        [
            (px_punta + dx * largo, py_punta + dy * largo),
            (px_punta - dy * ancho, py_punta + dx * ancho),
            (px_punta + dy * ancho, py_punta - dx * ancho),
        ],
        fill=color,
    )

    imagen = imagen.resize((lado, lado), Image.LANCZOS)

    return ctk.CTkImage(light_image=imagen, dark_image=imagen,
                        size=(lado, lado))


def completar_direccion(texto):
    """Convierte lo que se escriba a mano en una URL de video.

    Dos reglas, sin adivinar mas alla de eso:

    - Si ya trae esquema (http://...), se usa tal cual. Quien escribe
      una URL entera sabe lo que quiere y no hay que tocarsela.
    - Si no, se toma como host, con puerto opcional, y se le pone el
      puerto por defecto y la ruta /video de IP Webcam.

    >>> completar_direccion("192.168.0.25")
    'http://192.168.0.25:8080/video'
    >>> completar_direccion("192.168.0.25:4747")
    'http://192.168.0.25:4747/video'
    >>> completar_direccion("http://10.0.0.5:8080/videofeed")
    'http://10.0.0.5:8080/videofeed'
    """
    texto = texto.strip()

    if not texto:
        return ""

    if "://" in texto:
        return texto

    if ":" not in texto:
        texto = f"{texto}:{PUERTO_IP_WEBCAM}"

    return f"http://{texto}/video"


def primera_familia_disponible(candidatas):
    instaladas = set(tkfont.families())

    for familia in candidatas:
        if familia in instaladas:
            return familia

    return candidatas[-1]


# =================================
# Indices de landmarks (MediaPipe Pose, 33 puntos)
# =================================

OREJA_D = 8
HOMBRO_I = 11
HOMBRO_D = 12
CADERA_I = 23
CADERA_D = 24
RODILLA_D = 26
TOBILLO_D = 28


# =================================
# Umbrales
# =================================
# El indice de encorvamiento es la distancia perpendicular de la oreja
# a la recta cadera-hombro, dividida por el largo del tronco. Con las
# proporciones antropometricas estandar (Drillis & Contini) el tronco
# mide 0.288 de la estatura y el tramo hombro-oreja 0.112, o sea una
# razon d/L = 0.389. De ahi:
#
#       indice = 0.389 * sin(adelantamiento de la cabeza)
#
#   indice 0.07 ->  10 grados
#   indice 0.10 ->  15 grados
#   indice 0.13 ->  20 grados
#   indice 0.19 ->  30 grados   <- era el valor anterior
#
# 30 grados es encorvamiento severo: por eso avisaba tan tarde. 15 es
# donde ya se nota a simple vista.
#
# El 0.08 ya NO es una estimacion: sale de calibrar_encorvamiento.py
# con esta camara y este cuerpo (derecho 0.024, encorvado 0.140), y
# coincide con lo medido sobre fotos de referencia, donde ninguna
# espalda recta clara paso de 0.073 y ninguna encorvada bajo de 0.144.
# Dos fuentes independientes dejando el limite en la misma franja.
#
# Reserva: la separacion de esa toma fue de 2.0 desviaciones, justo el
# minimo que el script da por bueno. Si en uso real aparecen avisos
# falsos, repetir la medida bien de perfil antes de tocar el numero.
UMBRAL_INCLINACION = 20.0
UMBRAL_JOROBA = 0.08

# Histeresis: al bajar el umbral, los valores rondando el limite
# encienden y apagan la alerta varias veces por segundo y el campo de
# color parpadea. Con margen, encender cuesta mas que apagar.
MARGEN_JOROBA = 0.02
MARGEN_INCLINACION = 3.0

# Fotogramas que se promedian al medir el perfil. Con uno solo, el
# temblor de los landmarks quedaria grabado como si fuera la medida.
CAPTURAS_PARA_PERFIL = 30


# =================================
# Conexiones del skeleton
# =================================

POSE_CONNECTIONS = [
    (0, 1), (0, 4),
    (1, 2), (2, 3), (3, 7),
    (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21),
    (17, 19),
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (27, 31),
    (28, 32),
    (29, 31),
    (30, 32),
]


# =================================
# Geometria (igual que pose_angulos_sentadilla.py)
# =================================

def calcular_angulo(a, b, c):
    ang_ba = math.atan2(a[1] - b[1], a[0] - b[0])
    ang_bc = math.atan2(c[1] - b[1], c[0] - b[0])

    angulo = math.degrees(ang_ba - ang_bc)
    angulo = abs(angulo)

    if angulo > 180:
        angulo = 360 - angulo

    return angulo


def punto_medio(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def angulo_respecto_vertical(punto_inferior, punto_superior):
    dx = punto_superior[0] - punto_inferior[0]
    dy = punto_superior[1] - punto_inferior[1]
    return math.degrees(math.atan2(dx, -dy))


def indice_encorvamiento(oreja, hombro_medio, cadera_medio):
    ax, ay = cadera_medio
    bx, by = hombro_medio
    px, py = oreja

    largo_tronco = math.hypot(bx - ax, by - ay)

    if largo_tronco < 1e-6:
        return 0.0

    distancia_perpendicular = (
        (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    ) / largo_tronco

    return distancia_perpendicular / largo_tronco


class LectorCamara:
    """Lee la camara en un hilo aparte, guardando SOLO el frame mas
    reciente.

    Sin esto, cv2.VideoCapture acumula los frames que llegan por red en
    una cola interna. Como MediaPipe tarda mas en procesar un frame de
    lo que tarda en llegar el siguiente, esa cola crece sin parar y el
    video que se ve queda cada vez mas atrasado respecto a la realidad
    (llega a minutos). Descartando los frames viejos, siempre se procesa
    lo que esta pasando ahora.
    """

    def __init__(self, fuente):
        self.fuente = fuente

        self.frame = None
        self.candado = threading.Lock()

        self.corriendo = False
        self.conectado = False
        self.error = None
        self.ultimo_frame_en = 0.0

        self.hilo = None

    def iniciar(self):
        self.corriendo = True
        self.ultimo_frame_en = time.time()
        self.hilo = threading.Thread(target=self._bucle, daemon=True)
        self.hilo.start()

    def _abrir(self):
        """Abre la fuente. Para URLs se fijan timeouts explicitos: sin
        ellos, si el celular no esta transmitiendo, OpenCV se queda
        esperando indefinidamente."""

        if isinstance(self.fuente, str):
            captura = cv2.VideoCapture(
                self.fuente,
                cv2.CAP_FFMPEG,
                [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, TIMEOUT_CONEXION_MS,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, TIMEOUT_LECTURA_MS,
                ],
            )
        else:
            captura = cv2.VideoCapture(self.fuente)

        try:
            captura.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        return captura

    def _bucle(self):
        """Todo el trabajo bloqueante (abrir, leer, liberar) ocurre en
        este hilo. La interfaz nunca se queda congelada esperandolo."""

        captura = None

        try:
            captura = self._abrir()

            if not captura.isOpened():
                self.error = f"No se pudo conectar con {self.fuente}"
                return

            self.conectado = True
            self.ultimo_frame_en = time.time()

            while self.corriendo:
                ok, frame = captura.read()

                if not ok:
                    self.error = "Se perdio la senal de la camara"
                    break

                # Sobrescribe el anterior: si no alcanzo a procesarse,
                # se descarta a proposito.
                with self.candado:
                    self.frame = frame
                    self.ultimo_frame_en = time.time()

        except Exception as e:
            self.error = f"Error leyendo la camara: {e}"

        finally:
            self.conectado = False
            self.corriendo = False

            # Se libera desde el propio hilo: hacerlo desde la interfaz
            # la bloquearia si read() quedo colgado.
            if captura is not None:
                captura.release()

    def leer(self):
        with self.candado:
            frame = self.frame
            self.frame = None

        return (frame is not None), frame

    def segundos_sin_frame(self):
        return time.time() - self.ultimo_frame_en

    def detener(self):
        """Solo baja la bandera. No se hace join ni release aqui para no
        congelar la interfaz si el hilo esta atascado en read(); el hilo
        es daemon y se limpia solo."""

        self.corriendo = False


def clasificar_postura(angulo_inclinacion, indice_joroba,
                       alerta_previa=False):
    """Devuelve (texto, hay_alerta).

    El booleano se usa para resaltar la postura en la interfaz; el
    texto por si solo ya dice que pasa, el color solo lo refuerza.

    `alerta_previa` aplica la histeresis: si ya habia alerta, hace
    falta bajar mas para apagarla.

    >>> clasificar_postura(0.0, 0.05)
    ('Espalda recta', False)
    >>> clasificar_postura(0.0, 0.12)
    ('Posible encorvamiento', True)

    En la franja entre el umbral y el margen, lo que decide es si ya
    venia encendida: 0.07 mantiene la alerta pero no la enciende.

    >>> clasificar_postura(0.0, 0.07, alerta_previa=True)[1]
    True
    >>> clasificar_postura(0.0, 0.07, alerta_previa=False)[1]
    False
    """
    margen_j = MARGEN_JOROBA if alerta_previa else 0.0
    margen_i = MARGEN_INCLINACION if alerta_previa else 0.0

    etiquetas = []

    if abs(angulo_inclinacion) > UMBRAL_INCLINACION - margen_i:
        lado = "adelante" if angulo_inclinacion > 0 else "atras"
        etiquetas.append(f"Inclinado {lado}")

    if abs(indice_joroba) > UMBRAL_JOROBA - margen_j:
        etiquetas.append("Posible encorvamiento")

    if not etiquetas:
        return "Espalda recta", False

    return " | ".join(etiquetas), True


# =================================
# Aplicacion
# =================================

class PoseApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("Correccion de ejercicio - Prototipo PC")
        self.geometry("1100x700")
        self.minsize(900, 560)

        # Arranca ocupando toda la pantalla. Se usa el estado
        # "maximizado" de Windows en vez de -fullscreen porque Tk
        # reporta la resolucion ya reducida por el escalado de DPI
        # (1536x864 en una pantalla de 1920x1080) y -fullscreen se
        # queda con ese tamano, dejando franjas sin cubrir.
        # F11 da pantalla completa real (sin barra de titulo) y Escape
        # vuelve a la ventana normal.
        self.pantalla_completa = False
        self._ya_maximizada = False
        # Se engancha a <Map> y no a after(0): CustomTkinter aplica su
        # propia geometria en un callback diferido posterior, que
        # deshacia el maximizado y devolvia la ventana a 1100x700.
        self.bind("<Map>", self._maximizar_al_abrir)
        self.bind("<F11>", self._alternar_pantalla_completa)
        self.bind("<Escape>", self._salir_pantalla_completa)

        ctk.set_appearance_mode("dark")
        self.configure(fg_color=COLOR["lienzo"])

        self.cap = None
        self.detector = None
        self.corriendo = False

        self.start_time = 0.0
        self.previous_time = 0.0
        self.fps_medido = 0.0
        self.alerta_previa = False

        # Buzon de un solo hueco entre el hilo de proceso y el de la
        # interfaz: solo interesa el ultimo frame, los viejos sobran.
        self.ultima_salida = None
        self.candado_salida = threading.Lock()
        self.hilo_proceso = None
        self.senal_perdida = False
        self.motivo_corte = None

        # Pasa a True con el primer frame pintado: sirve para cambiar
        # el estado de "Conectando" a "Midiendo" una sola vez.
        self.midiendo = False

        # Identifica la sesion de medida en curso. _pintar se reagenda
        # solo con after(); si se para y se vuelve a empezar antes de
        # que salte el after pendiente, ese callback reviviria y
        # quedarian dos cadenas pintando a la vez. Cada cadena lleva su
        # numero y la que no coincide se apaga.
        self.sesion = 0

        # El hilo de proceso no puede consultar los botones: Tk solo
        # admite llamadas desde el hilo principal. La interfaz deja
        # aqui la constante de giro ya resuelta y el hilo la lee.
        self.giro_grados = 0
        self.rotacion_actual = None

        # Cuenta y juzga las repeticiones. Vive en el hilo de la
        # interfaz porque necesita la alerta de postura, que se decide
        # al mostrar las medidas, y es calculo de cuatro comparaciones.
        self.contador = ContadorSentadillas()

        # Perfil de la persona, si ya se midio en otra sesion. Sirve
        # para descartar fotogramas donde los landmarks se fueron.
        self.perfil = PerfilCorporal.cargar()

        # Mientras esto tenga una lista, el hilo de proceso va
        # guardando capturas para construir el perfil.
        self.capturas_perfil = None
        self.estatura_pedida = 170.0
        self.frames_descartados = 0

        # Lectura del QR y red a la que volver despues: la del celular
        # no comparte internet, asi que no conviene quedarse ahi.
        self.escaner = None
        self.red_previa = None

        # Buzones del hilo que entra a la red del celular: escribe
        # aqui y el hilo de la interfaz lo recoge.
        self.qr_resultado = None
        self.qr_avance = None

        # PhotoImage que se reutiliza entre frames.
        self.imagen_tk = None

        self._construir_fuentes()
        self._construir_ui()

    def _construir_fuentes(self):
        # CTkFont necesita que la ventana raiz ya exista, por eso se
        # crean aqui y no como constantes de modulo.
        titulo = primera_familia_disponible(FAMILIAS_TITULO)
        texto = primera_familia_disponible(FAMILIAS_TEXTO)
        cifra = primera_familia_disponible(FAMILIAS_CIFRA)

        self.fuente_seccion = ctk.CTkFont(family=titulo, size=13)
        self.fuente_texto = ctk.CTkFont(family=texto, size=14)
        self.fuente_boton = ctk.CTkFont(family=titulo, size=15)

        # El veredicto se lee a varios metros, de reojo y en
        # movimiento: por eso es lo unico que va a este tamano.
        self.fuente_veredicto = ctk.CTkFont(family=titulo, size=34)

        # Los angulos se leen en una hoja de medidas, no en tarjetas.
        # Van alineados a la derecha para que los digitos no se
        # desplacen al cambiar de dos a tres cifras.
        self.fuente_medida = ctk.CTkFont(family=cifra, size=26)
        self.fuente_etiqueta = ctk.CTkFont(family=texto, size=15)
        self.fuente_pie = ctk.CTkFont(family=texto, size=12)

    def _maximizar_al_abrir(self, evento=None):
        """Maximiza una sola vez, cuando la ventana ya esta en pantalla.

        Solo la primera vez: si no, restaurar la ventana desde la barra
        de tareas la volveria a maximizar y no se podria dejar pequena.
        """
        if self._ya_maximizada:
            return

        self._ya_maximizada = True
        self.after_idle(lambda: self.state("zoomed"))

    def _alternar_pantalla_completa(self, evento=None):
        self.pantalla_completa = not self.pantalla_completa
        self.attributes("-fullscreen", self.pantalla_completa)

    def _salir_pantalla_completa(self, evento=None):
        self.pantalla_completa = False
        self.attributes("-fullscreen", False)
        self.state("zoomed")

    # ---------------------------------
    # Construccion de la interfaz
    # ---------------------------------

    def _construir_ui(self):

        # El video manda en superficie; el panel es una columna fija a
        # la derecha, ancha para que los angulos entren grandes.
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0, minsize=420)
        self.grid_rowconfigure(0, weight=1)

        # --- Video (izquierda) ---
        # Sin esquinas redondeadas ni margen: es la imagen de un
        # instrumento, no una tarjeta.
        self.frame_video = ctk.CTkFrame(
            self, fg_color=COLOR["lienzo"], corner_radius=0,
        )
        self.frame_video.grid(row=0, column=0, sticky="nsew")
        self.frame_video.grid_columnconfigure(0, weight=1)
        self.frame_video.grid_rowconfigure(0, weight=1)

        self.label_video = ctk.CTkLabel(self.frame_video, text="")
        self.label_video.grid(row=0, column=0, sticky="nsew")

        self._video_box = (800, 600)
        self.frame_video.bind("<Configure>", self._on_video_resize)

        # --- Panel (derecha) ---
        panel = ctk.CTkFrame(
            self, width=420, fg_color=COLOR["superficie"], corner_radius=0,
        )
        panel.grid(row=0, column=1, sticky="nsew")
        panel.grid_propagate(False)
        panel.grid_columnconfigure(0, weight=1)
        # El sobrante se lo queda un hueco despues de las medidas, no
        # el veredicto: con weight en el campo de color, el texto
        # quedaba flotando en mitad de 300px de rojo.
        panel.grid_rowconfigure(4, weight=1)

        # ---- Controles: arriba y deliberadamente discretos ----
        controles = ctk.CTkFrame(panel, fg_color="transparent")
        controles.grid(row=0, column=0, sticky="ew",
                       padx=ESPACIO["lg"], pady=(ESPACIO["lg"], 0))
        controles.grid_columnconfigure(1, weight=1)

        self._etiqueta_control(controles, 0, "Camara")
        self.tipo_camara = ctk.CTkSegmentedButton(
            controles, values=["Esta PC", "Celular"],
            command=self._cambiar_tipo_camara,
            font=self.fuente_texto, corner_radius=0, border_width=1,
            # CTkSegmentedButton usa un unico text_color para las dos
            # opciones. Con texto claro, la activa tiene que seguir
            # siendo oscura: se distingue por ser mas clara que la
            # inactiva, no por invertir el texto.
            fg_color=COLOR["superficie"], unselected_color=COLOR["superficie"],
            unselected_hover_color=COLOR["campo"],
            selected_color=COLOR["filete"],
            selected_hover_color=COLOR["filete"],
            text_color=COLOR["texto"],
        )
        self.tipo_camara.set("Esta PC")
        self.tipo_camara.grid(row=0, column=1, sticky="ew", pady=ESPACIO["xs"])

        self._etiqueta_control(controles, 1, "Origen")
        self.frame_fuente = ctk.CTkFrame(controles, fg_color="transparent")
        self.frame_fuente.grid(row=1, column=1, sticky="ew",
                               pady=ESPACIO["xs"])

        self.frame_fuente_pc = ctk.CTkFrame(
            self.frame_fuente, fg_color="transparent"
        )
        self.entry_indice_pc = ctk.CTkEntry(
            self.frame_fuente_pc, font=self.fuente_texto, corner_radius=0,
            fg_color=COLOR["campo"], border_color=COLOR["filete"],
            text_color=COLOR["texto"], border_width=1,
        )
        self.entry_indice_pc.insert(0, "0")
        self.entry_indice_pc.pack(fill="x")

        # --- Opcion: celular ---
        # Dos formas de dar la direccion. La corta sirve cuando el
        # celular esta en la red de siempre y solo cambia el ultimo
        # numero; la completa, cuando se cambia de red o de puerto.
        self.frame_fuente_celular = ctk.CTkFrame(
            self.frame_fuente, fg_color="transparent"
        )

        self.modo_direccion = ctk.CTkSegmentedButton(
            self.frame_fuente_celular,
            values=["Ultimo numero", "Direccion completa"],
            command=self._cambiar_modo_direccion,
            font=self.fuente_texto, corner_radius=0, border_width=1,
            fg_color=COLOR["superficie"], unselected_color=COLOR["superficie"],
            unselected_hover_color=COLOR["campo"],
            selected_color=COLOR["filete"],
            selected_hover_color=COLOR["filete"],
            text_color=COLOR["texto"],
        )
        self.modo_direccion.set("Ultimo numero")
        self.modo_direccion.pack(fill="x", pady=(0, ESPACIO["xs"]))

        self.frame_ip_corta = ctk.CTkFrame(
            self.frame_fuente_celular, fg_color="transparent"
        )
        ctk.CTkLabel(
            self.frame_ip_corta, text=PREFIJO_IP_CELULAR,
            font=self.fuente_texto, text_color=COLOR["tenue"],
        ).pack(side="left")
        self.entry_ip_final = ctk.CTkEntry(
            self.frame_ip_corta, width=58, font=self.fuente_texto,
            corner_radius=0, fg_color=COLOR["campo"],
            border_color=COLOR["filete"], text_color=COLOR["texto"],
            border_width=1,
        )
        self.entry_ip_final.insert(0, "8")
        self.entry_ip_final.pack(side="left", padx=ESPACIO["xs"])
        ctk.CTkLabel(
            self.frame_ip_corta, text=":" + PUERTO_IP_WEBCAM,
            font=self.fuente_texto, text_color=COLOR["tenue"],
        ).pack(side="left")

        self.entry_ip_completa = ctk.CTkEntry(
            self.frame_fuente_celular, font=self.fuente_texto,
            corner_radius=0, fg_color=COLOR["campo"],
            border_color=COLOR["filete"], text_color=COLOR["texto"],
            placeholder_text_color=COLOR["tenue"], border_width=1,
            placeholder_text="192.168.0.25  o  http://...:8080/video",
        )

        self.frame_ip_corta.pack(fill="x")

        # Atajo para no teclear nada: Camcel muestra un QR con su
        # direccion y, si creo su propia red, tambien con la clave.
        self.boton_qr = ctk.CTkButton(
            self.frame_fuente_celular, text="Leer QR del celular",
            command=self._leer_qr, height=32,
            font=self.fuente_texto, corner_radius=0,
            fg_color="transparent", hover_color=COLOR["campo"],
            border_width=1, border_color=COLOR["filete"],
            text_color=COLOR["texto"],
            text_color_disabled=COLOR["filete"],
        )
        self.boton_qr.pack(fill="x", pady=(ESPACIO["xs"], 0))

        # La red del celular no da internet. Sin esta salida, el unico
        # regreso a la WiFi de siempre seria cerrar la aplicacion.
        self.boton_volver_red = ctk.CTkButton(
            self.frame_fuente_celular, text="", command=self._volver_a_mi_red,
            height=32, font=self.fuente_texto, corner_radius=0,
            fg_color="transparent", hover_color=COLOR["campo"],
            border_width=1, border_color=COLOR["filete"],
            text_color=COLOR["texto"],
        )

        self.frame_fuente_pc.pack(fill="x")

        self._etiqueta_control(controles, 2, "Giro")

        giro = ctk.CTkFrame(controles, fg_color="transparent")
        giro.grid(row=2, column=1, sticky="ew", pady=ESPACIO["xs"])

        # Las imagenes se guardan en el objeto: si solo viven dentro de
        # este metodo, el recolector se las lleva y los botones salen
        # vacios.
        self.icono_antihorario = crear_icono_giro(horario=False)
        self.icono_horario = crear_icono_giro(horario=True)

        for columna, (icono, paso) in enumerate(
            ((self.icono_antihorario, -90), (self.icono_horario, 90))
        ):
            ctk.CTkButton(
                giro, text="", image=icono, width=46, height=32,
                corner_radius=0, fg_color=COLOR["campo"],
                hover_color=COLOR["filete"],
                border_width=1, border_color=COLOR["filete"],
                command=lambda p=paso: self._girar(p),
            ).grid(row=0, column=columna, padx=(0, ESPACIO["sm"]))

        # El angulo actual se escribe al lado: sin el, tras varios
        # clics no hay forma de saber en que posicion quedo.
        self.label_giro = ctk.CTkLabel(
            giro, text="0°", font=self.fuente_texto,
            text_color=COLOR["tenue"], anchor="w",
        )
        self.label_giro.grid(row=0, column=2, padx=(ESPACIO["sm"], 0))

        self._etiqueta_control(controles, 3, "Estatura")

        perfil_fila = ctk.CTkFrame(controles, fg_color="transparent")
        perfil_fila.grid(row=3, column=1, sticky="ew", pady=ESPACIO["xs"])

        self.entry_estatura = ctk.CTkEntry(
            perfil_fila, width=58, font=self.fuente_texto, corner_radius=0,
            fg_color=COLOR["campo"], border_color=COLOR["filete"],
            text_color=COLOR["texto"], border_width=1,
        )
        self.entry_estatura.insert(
            0, f"{self.perfil.estatura_cm:.0f}" if self.perfil else "170"
        )
        self.entry_estatura.pack(side="left")

        ctk.CTkLabel(
            perfil_fila, text="cm", font=self.fuente_texto,
            text_color=COLOR["tenue"],
        ).pack(side="left", padx=ESPACIO["xs"])

        self.boton_perfil = ctk.CTkButton(
            perfil_fila, text="Medir perfil", command=self._medir_perfil,
            height=28, font=self.fuente_texto, corner_radius=0,
            fg_color="transparent", hover_color=COLOR["campo"],
            border_width=1, border_color=COLOR["filete"],
            text_color=COLOR["texto"],
            text_color_disabled=COLOR["filete"],
        )
        self.boton_perfil.pack(side="left", padx=(ESPACIO["sm"], 0))

        botones = ctk.CTkFrame(panel, fg_color="transparent")
        botones.grid(row=1, column=0, sticky="ew",
                     padx=ESPACIO["lg"], pady=ESPACIO["md"])
        botones.grid_columnconfigure(0, weight=1)
        botones.grid_columnconfigure(1, weight=1)

        self.boton_iniciar = ctk.CTkButton(
            botones, text="Empezar", command=self.iniciar, height=46,
            font=self.fuente_boton, corner_radius=0,
            fg_color=COLOR["texto"], hover_color="#FFFFFF",
            text_color=COLOR["superficie"],
            text_color_disabled=COLOR["filete"],
        )
        self.boton_iniciar.grid(row=0, column=0, sticky="ew",
                                padx=(0, ESPACIO["xs"]))

        self.boton_detener = ctk.CTkButton(
            botones, text="Parar", command=self.detener, height=46,
            font=self.fuente_boton, corner_radius=0,
            fg_color="transparent", hover_color=COLOR["campo"],
            border_width=1, border_color=COLOR["filete"],
            text_color=COLOR["texto"],
            text_color_disabled=COLOR["filete"],
        )
        self.boton_detener.grid(row=0, column=1, sticky="ew",
                                padx=(ESPACIO["xs"], 0))

        # ---- Veredicto: el unico elemento con volumen ----
        # Un campo de color a todo el ancho, no una etiqueta pequena.
        # La vision periferica capta una superficie de color; no capta
        # un texto de 20px a tres metros.
        self.campo_veredicto = ctk.CTkFrame(
            panel, fg_color=COLOR["filete"], corner_radius=0, height=160,
        )
        self.campo_veredicto.grid(row=2, column=0, sticky="nsew")
        self.campo_veredicto.grid_propagate(False)
        self.campo_veredicto.grid_columnconfigure(0, weight=1)
        self.campo_veredicto.grid_rowconfigure(0, weight=1)

        self.label_veredicto = ctk.CTkLabel(
            self.campo_veredicto, text="Sin lectura",
            font=self.fuente_veredicto, text_color=COLOR["tenue"],
            wraplength=360, justify="left", anchor="w",
        )
        self.label_veredicto.grid(row=0, column=0, sticky="nsew",
                                  padx=ESPACIO["lg"], pady=ESPACIO["md"])

        # ---- Medidas: hoja de datos, no tarjetas ----
        medidas = ctk.CTkFrame(panel, fg_color="transparent")
        medidas.grid(row=3, column=0, sticky="ew",
                     padx=ESPACIO["lg"], pady=(ESPACIO["md"], 0))
        medidas.grid_columnconfigure(0, weight=1)
        medidas.grid_columnconfigure(1, weight=0)

        self.valores = {}

        filas = [
            ("repeticiones", "Repeticiones"),
            ("rodilla", "Rodilla"),
            ("cadera", "Cadera"),
            ("inclinacion", "Tronco"),
            ("joroba", "Encorvamiento"),
        ]

        for indice, (clave, etiqueta) in enumerate(filas):
            fila = indice * 2

            ctk.CTkLabel(
                medidas, text=etiqueta, font=self.fuente_etiqueta,
                text_color=COLOR["tenue"], anchor="w",
            ).grid(row=fila, column=0, sticky="w", pady=ESPACIO["xs"])

            valor = ctk.CTkLabel(
                medidas, text="—", font=self.fuente_medida,
                text_color=COLOR["texto"], anchor="e",
            )
            valor.grid(row=fila, column=1, sticky="e", pady=ESPACIO["xs"])
            self.valores[clave] = valor

            # Filete entre filas: separa datos tabulares, que es
            # informacion, no adorno. La ultima fila no lo lleva.
            if indice < len(filas) - 1:
                ctk.CTkFrame(
                    medidas, height=1, fg_color=COLOR["filete"],
                    corner_radius=0,
                ).grid(row=fila + 1, column=0, columnspan=2, sticky="ew")

        # Hueco elastico: empuja el pie al fondo del panel.
        ctk.CTkFrame(panel, fg_color="transparent", height=1).grid(
            row=4, column=0, sticky="nsew"
        )

        # ---- Pie: estado a la izquierda, ritmo a la derecha ----
        pie = ctk.CTkFrame(panel, fg_color="transparent")
        pie.grid(row=5, column=0, sticky="ew",
                 padx=ESPACIO["lg"], pady=ESPACIO["lg"])
        pie.grid_columnconfigure(0, weight=1)

        self.label_estado = ctk.CTkLabel(
            pie, text="Detenido", font=self.fuente_pie,
            text_color=COLOR["tenue"], anchor="w", justify="left",
            wraplength=270,
        )
        self.label_estado.grid(row=0, column=0, sticky="w")

        self.valores["fps"] = ctk.CTkLabel(
            pie, text="", font=self.fuente_pie,
            text_color=COLOR["tenue"], anchor="e",
        )
        self.valores["fps"].grid(row=0, column=1, sticky="e")

        self._actualizar_botones()

    def _etiqueta_control(self, padre, fila, texto):
        """Etiqueta en caja baja, a la izquierda de su control.

        Antes iba encima y en mayusculas: tres etiquetas apiladas que
        empujaban los controles hacia abajo sin aportar nada. Compartir
        fila con el campo ahorra alto y las relaciona sin ruido.
        """
        ctk.CTkLabel(
            padre, text=texto, font=self.fuente_seccion,
            text_color=COLOR["tenue"], anchor="w", width=78,
        ).grid(row=fila, column=0, sticky="w", padx=(0, ESPACIO["sm"]))

    def _actualizar_botones(self):
        """Deja habilitado solo el boton que hace algo ahora mismo.

        Antes los dos estaban siempre activos e 'Iniciar' salia en
        silencio si ya se estaba ejecutando: el click no daba ninguna
        respuesta.
        """
        self.boton_iniciar.configure(
            state="disabled" if self.corriendo else "normal"
        )
        self.boton_detener.configure(
            state="normal" if self.corriendo else "disabled"
        )

    def _set_estado(self, texto, color=None):
        self.label_estado.configure(
            text=texto, text_color=color or COLOR["tenue"]
        )

    def _set_veredicto(self, texto, fondo, tinte):
        """Pinta el campo de veredicto entero, no solo el texto."""
        self.campo_veredicto.configure(fg_color=fondo)
        self.label_veredicto.configure(text=texto, text_color=tinte)

    def _girar(self, paso):
        """Cada clic gira 90 grados, igual que voltear el celular.

        Se aplica en caliente: el hilo de proceso lee el valor en cada
        frame, asi que no hace falta parar y volver a empezar para
        enderezar la imagen.
        """
        self.giro_grados = (self.giro_grados + paso) % 360
        self.rotacion_actual = ROTACIONES[self.giro_grados]
        self.label_giro.configure(text=f"{self.giro_grados}°")

    # ---------------------------------
    # Conexion por QR
    # ---------------------------------

    def _leer_qr(self):
        """Escanea el QR de Camcel y deja la direccion lista.

        Todo el trabajo lento (abrir la webcam, decodificar, entrar a la
        red del celular) corre fuera del hilo de la interfaz; aqui solo
        se lanza y se revisa el avance con after().
        """

        if self.corriendo:
            self._set_estado(
                "Para la medicion antes de leer el QR: la camara esta ocupada.",
                COLOR["alerta"],
            )
            return

        indice = 0
        texto_indice = self.entry_indice_pc.get().strip()

        if texto_indice.isdigit():
            indice = int(texto_indice)

        self.escaner = conexion_qr.EscanerQr(indice_camara=indice)
        self.escaner.iniciar()

        self.boton_qr.configure(state="disabled")
        self._set_estado("Apunta la camara del PC al QR del celular.")
        self._set_veredicto("Buscando QR", COLOR["filete"], COLOR["tenue"])
        self._revisar_qr()

    def _revisar_qr(self):

        escaner = self.escaner

        if escaner is None:
            return

        self._pintar_busqueda_qr(escaner)

        if escaner.datos is not None:
            self._aplicar_qr(escaner.datos)
            return

        if escaner.terminado:
            self.escaner = None
            self.boton_qr.configure(state="normal")
            self._limpiar_video()
            self._set_estado(
                escaner.error or "No se pudo leer el QR.", COLOR["alerta"]
            )
            return

        # Mas seguido que el sondeo de antes: esto ya no solo vigila,
        # tambien dibuja, y a 80 ms el video se veia a saltos.
        self.after(MS_ENTRE_PINTADOS, self._revisar_qr)

    def _pintar_busqueda_qr(self, escaner):
        """Muestra la webcam mientras busca, para saber a donde apuntar."""

        frame, esquinas = escaner.ultimo_frame()

        if frame is None:
            return

        vista = frame.copy()

        if esquinas is not None:
            # Marcar el QR en cuanto se ve, aunque aun no se haya
            # podido decodificar: confirma que la camara lo tiene
            # encuadrado y que solo falta acercarse o estabilizar.
            puntos = esquinas.astype(int).reshape(-1, 2)
            cv2.polylines(vista, [puntos], True, BGR_TRONCO, 3)

        self._mostrar_frame(vista)

    def _limpiar_video(self):
        """Deja el panel de video vacio al salir del escaneo."""
        self.imagen_tk = None
        self.label_video.configure(image=None)

    def _aplicar_qr(self, datos):
        """Usa lo que traia el QR: red si hace falta, y la direccion."""

        self.escaner = None
        self.qr_resultado = None
        self.qr_avance = None

        # Entrar a la red del celular bloquea varios segundos, asi que
        # va en un hilo. Ese hilo solo deja escrito como va: llamar a
        # after() o tocar widgets desde aqui revienta, porque Tk solo
        # admite llamadas desde el hilo principal.
        def trabajo():
            try:
                if "ssid" in datos and "clave" in datos:
                    anterior = conexion_qr.conectar_a_red(
                        datos["ssid"], datos["clave"],
                        al_avanzar=lambda t: setattr(self, "qr_avance", t),
                    )

                    # Solo se apunta la PRIMERA red de la que se salio.
                    # Al escanear otra vez ya estando en la del celular,
                    # "la anterior" seria esa misma, y al cerrar la app
                    # volveria a dejar el PC sin internet.
                    if self.red_previa is None and anterior != datos["ssid"]:
                        self.red_previa = anterior

                self.qr_resultado = ("listo", datos)

            except conexion_qr.ErrorQr as e:
                self.qr_resultado = ("fallo", str(e))

        threading.Thread(target=trabajo, daemon=True).start()
        self._revisar_conexion_qr()

    def _revisar_conexion_qr(self):
        """Traslada a la interfaz lo que va dejando el hilo de conexion."""

        if self.qr_avance is not None:
            self._set_estado(self.qr_avance)
            self.qr_avance = None

        if self.qr_resultado is None:
            self.after(100, self._revisar_conexion_qr)
            return

        clase, carga = self.qr_resultado
        self.qr_resultado = None

        if clase == "listo":
            self._qr_listo(carga)
        elif clase == "red_restaurada":
            self._red_restaurada(carga)
        else:
            self._qr_fallo(carga)

    def _qr_listo(self, datos):
        """Deja la direccion escrita y lista para empezar."""

        self.tipo_camara.set("Celular")
        self._cambiar_tipo_camara("Celular")

        self.modo_direccion.set("Direccion completa")
        self._cambiar_modo_direccion("Direccion completa")

        self.entry_ip_completa.delete(0, "end")
        self.entry_ip_completa.insert(0, conexion_qr.url_de(datos))

        self.boton_qr.configure(state="normal")
        self._limpiar_video()
        self._set_veredicto("Sin lectura", COLOR["filete"], COLOR["tenue"])
        self._actualizar_boton_red()
        self._set_estado("Direccion lista. Pulsa Empezar.", COLOR["correcto"])

    def _qr_fallo(self, mensaje):
        self.boton_qr.configure(state="normal")
        self._limpiar_video()
        self._set_veredicto("Sin lectura", COLOR["filete"], COLOR["tenue"])
        self._set_estado(mensaje, COLOR["alerta"])

    def _actualizar_boton_red(self):
        """El boton de regreso solo existe si hay a donde volver."""

        if not self.red_previa:
            self.boton_volver_red.pack_forget()
            return

        self.boton_volver_red.configure(
            text=f"Volver a {self.red_previa}"
        )
        self.boton_volver_red.pack(fill="x", pady=(ESPACIO["xs"], 0))

    def _volver_a_mi_red(self):
        """Regresa a la WiFi de siempre, sin cerrar la aplicacion."""

        destino = self.red_previa

        if not destino:
            return

        if self.corriendo:
            self.detener()

        self.boton_volver_red.configure(state="disabled")
        self._set_estado(f"Volviendo a {destino}...")

        def trabajo():
            conexion_qr.volver_a_red(destino)
            self.qr_resultado = ("red_restaurada", destino)

        threading.Thread(target=trabajo, daemon=True).start()
        self._revisar_conexion_qr()

    def _red_restaurada(self, destino):
        self.red_previa = None
        self.boton_volver_red.configure(state="normal")
        self._actualizar_boton_red()
        self._set_estado(f"De vuelta en {destino}.", COLOR["correcto"])

    def _cambiar_modo_direccion(self, seleccion):

        if seleccion == "Ultimo numero":
            self.entry_ip_completa.pack_forget()
            self.frame_ip_corta.pack(fill="x")
        else:
            self.frame_ip_corta.pack_forget()
            self.entry_ip_completa.pack(fill="x")

    def _cambiar_tipo_camara(self, seleccion):

        if seleccion == "Esta PC":
            self.frame_fuente_celular.pack_forget()
            self.frame_fuente_pc.pack(fill="x")
        else:
            self.frame_fuente_pc.pack_forget()
            self.frame_fuente_celular.pack(fill="x")

    def _on_video_resize(self, event):
        self._video_box = (event.width, event.height)

    def _obtener_fuente(self):

        if self.tipo_camara.get() == "Esta PC":
            texto = self.entry_indice_pc.get().strip()
            return int(texto) if texto.isdigit() else 0

        if self.modo_direccion.get() == "Ultimo numero":
            ultimo = self.entry_ip_final.get().strip()

            # Sin validar aqui, un campo vacio armaba
            # "http://192.168.1.:8080/video": una URL rota que pasaba la
            # comprobacion de cadena vacia y, cinco segundos despues,
            # culpaba al celular de un error que estaba en el formulario.
            if not (ultimo.isdigit() and 0 <= int(ultimo) <= 255):
                return ""

            return f"http://{PREFIJO_IP_CELULAR}{ultimo}:{PUERTO_IP_WEBCAM}/video"

        return completar_direccion(self.entry_ip_completa.get())

    def iniciar(self):

        if self.corriendo:
            return

        fuente = self._obtener_fuente()

        if fuente == "":
            self._set_estado(
                "Revisa la direccion del celular antes de empezar.",
                COLOR["alerta"],
            )
            return

        # Conectar puede tardar segundos (o fallar). Se hace dentro del
        # hilo del lector para que la ventana no se congele esperando;
        # el resultado se descubre mas adelante, en _pintar.
        self.cap = LectorCamara(fuente)
        self.cap.iniciar()

        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=POSE_MODEL_PATH
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.detector = vision.PoseLandmarker.create_from_options(options)

        self.start_time = time.time()
        self.previous_time = time.time()
        self.senal_perdida = False
        self.motivo_corte = None
        self.midiendo = False
        self.ultima_salida = None
        self.contador.reiniciar()

        self.corriendo = True
        self.sesion += 1

        self.hilo_proceso = threading.Thread(
            target=self._bucle_proceso, daemon=True
        )
        self.hilo_proceso.start()

        self._set_estado("Conectando")
        self._set_veredicto(
            "Buscando", COLOR["filete"], COLOR["tenue"]
        )
        self._actualizar_botones()

        self._pintar(self.sesion)

    def detener(self):

        self.corriendo = False

        # Se espera al hilo antes de soltar camara y detector: si no,
        # podria seguir usandolos justo despues de cerrarlos.
        if self.hilo_proceso is not None:
            self.hilo_proceso.join(timeout=1.0)
            self.hilo_proceso = None

        self.imagen_tk = None

        if self.cap is not None:
            self.cap.detener()
            self.cap = None

        if self.detector is not None:
            self.detector.close()
            self.detector = None

        self._reiniciar_valores()
        self._set_estado("Detenido")
        self._actualizar_botones()

    def _consejo_para(self, motivo):
        """El mensaje dice que hacer, no solo que fallo."""

        if self.tipo_camara.get() == "Esta PC":
            if "No se pudo conectar" in motivo:
                return "No hay camara en ese indice. Prueba con 1."
            return motivo

        if "No se pudo conectar" in motivo:
            return (
                "El celular no responde. Revisa que Camcel este "
                "transmitiendo y en la misma red WiFi."
            )

        if "Dejaron de llegar" in motivo:
            return (
                "Se corto la transmision. Revisa que Camcel siga "
                "abierto y la pantalla encendida."
            )

        return motivo

    def _reiniciar_valores(self):
        """Al detener, los ultimos numeros dejan de ser ciertos."""
        for clave, valor_label in self.valores.items():
            if clave == "fps":
                texto = ""
            elif clave == "repeticiones":
                # Las repeticiones hechas si siguen siendo ciertas: es
                # el resultado de la serie, no una lectura en vivo.
                texto = str(self.contador.repeticiones)
            else:
                texto = "—"

            valor_label.configure(text=texto)

        self._set_veredicto(
            "Sin lectura", COLOR["filete"], COLOR["tenue"]
        )

    def on_closing(self):
        self.detener()

        if self.escaner is not None:
            self.escaner.cancelar()

        # La red del celular no da internet: dejar el PC ahi al cerrar
        # seria dejarlo incomunicado sin avisar.
        if self.red_previa:
            conexion_qr.volver_a_red(self.red_previa)

        self.destroy()

    # ---------------------------------
    # Bucle principal (dirigido por eventos, no bloqueante)
    # ---------------------------------

    # ---------------------------------
    # Procesado (hilo aparte) y pintado (hilo de la interfaz)
    # ---------------------------------

    def _bucle_proceso(self):
        """Captura e inferencia, fuera del hilo de la interfaz.

        Antes esto corria dentro de un after() de Tk: cada frame
        bloqueaba la ventana ~17 ms de MediaPipe mas ~16 ms de pintado,
        y cualquier tropiezo de la red congelaba la interfaz entera.
        Eso eran los tirones. Aqui el hilo solo toca numpy, OpenCV y
        MediaPipe; Tk se queda para el hilo principal, que es lo unico
        que Tk admite.
        """
        # Referencias propias del hilo. detener() pone self.cap y
        # self.detector a None tras un join acotado a un segundo: si ese
        # join expira (una vuelta lenta de MediaPipe basta) este hilo
        # seguiria vivo y leeria None.
        cap = self.cap
        detector = self.detector

        while self.corriendo:
            ok, frame_bgr = cap.leer()

            # El lector termino: no se pudo conectar, o la transmision
            # se corto. El motivo lo trae el propio lector.
            if not cap.corriendo:
                self.motivo_corte = cap.error
                self.senal_perdida = True
                return

            if not ok:
                # Sin imagenes por mucho tiempo: no se logro conectar, o
                # la app del celular se cerro sin avisar. El limite
                # cubre tambien la conexion, porque cv2.VideoCapture no
                # siempre respeta su propio timeout de apertura y sin
                # esto la espera seria eterna.
                if cap.segundos_sin_frame() > SEGUNDOS_SIN_FRAME_PARA_CORTAR:
                    self.motivo_corte = (
                        "Dejaron de llegar imagenes del celular"
                        if cap.conectado
                        else f"No se pudo conectar con {cap.fuente}"
                    )
                    self.senal_perdida = True
                    return

                time.sleep(0.002)
                continue

            rotacion = self.rotacion_actual

            if rotacion is not None:
                frame_bgr = cv2.rotate(frame_bgr, rotacion)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=frame_rgb)
            timestamp_ms = int((time.time() - self.start_time) * 1000)

            # Si pararon mientras se preparaba el frame, no se entra a
            # MediaPipe: detener() podria estar cerrando el detector.
            if not self.corriendo:
                return

            try:
                result = detector.detect_for_video(mp_image, timestamp_ms)
            except Exception:
                # Un timestamp repetido o un frame corrupto no deben
                # tumbar el hilo: se descarta y se sigue.
                continue

            salida, medidas = self._procesar(frame_bgr, result)

            ahora = time.time()
            self.fps_medido = 1.0 / max(1e-6, ahora - self.previous_time)
            self.previous_time = ahora

            with self.candado_salida:
                self.ultima_salida = (salida, medidas)

    def _procesar(self, output, result):
        """Dibuja el skeleton y calcula las medidas. Sin tocar Tk."""
        medidas = {}

        if not result.pose_landmarks:
            return output, medidas

        pose_landmarks = result.pose_landmarks[0]
        height, width = output.shape[:2]

        # El skeleton se dibuja sobre el frame original y luego ese
        # frame se escala para mostrarlo. Si el grosor fuera fijo, a
        # baja resolucion las lineas se verian gordisimas al ampliar.
        escala_dibujo = min(width, height) / 480.0
        grosor_linea = max(1, round(2 * escala_dibujo))
        radio_punto = max(1, round(3 * escala_dibujo))

        points = [None] * len(pose_landmarks)

        for index, landmark in enumerate(pose_landmarks):
            x = max(0, min(width - 1, int(landmark.x * width)))
            y = max(0, min(height - 1, int(landmark.y * height)))
            points[index] = (x, y)
            cv2.circle(output, (x, y), radio_punto, BGR_SKELETON, -1)

        for start, end in POSE_CONNECTIONS:
            if (
                start < len(points) and end < len(points)
                and points[start] is not None and points[end] is not None
            ):
                cv2.line(
                    output, points[start], points[end],
                    BGR_SKELETON, grosor_linea,
                )

        if self.capturas_perfil is not None:
            self.capturas_perfil.append(list(points))

        # Un fotograma con un hueso del doble de largo no es una
        # postura rara, es una deteccion perdida: medirla ensucia
        # todo lo que viene detras.
        if (
            self.perfil is not None
            and not self.perfil.deteccion_coherente(points)
        ):
            self.frames_descartados += 1
            return output, medidas

        necesarios_piernas = [HOMBRO_D, CADERA_D, RODILLA_D, TOBILLO_D]

        if all(points[i] is not None for i in necesarios_piernas):
            medidas["rodilla"] = calcular_angulo(
                points[CADERA_D], points[RODILLA_D], points[TOBILLO_D]
            )
            medidas["cadera"] = calcular_angulo(
                points[HOMBRO_D], points[CADERA_D], points[RODILLA_D]
            )

        necesarios_espalda = [HOMBRO_I, HOMBRO_D, CADERA_I, CADERA_D, OREJA_D]

        if all(points[i] is not None for i in necesarios_espalda):
            hombro_medio = punto_medio(points[HOMBRO_I], points[HOMBRO_D])
            cadera_medio = punto_medio(points[CADERA_I], points[CADERA_D])

            medidas["inclinacion"] = angulo_respecto_vertical(
                cadera_medio, hombro_medio
            )
            medidas["joroba"] = indice_encorvamiento(
                points[OREJA_D], hombro_medio, cadera_medio
            )

            cv2.line(
                output,
                (int(cadera_medio[0]), int(cadera_medio[1])),
                (int(hombro_medio[0]), int(hombro_medio[1])),
                BGR_TRONCO, grosor_linea,
            )

        return output, medidas

    def _pintar(self, sesion):
        """Pinta el ultimo resultado disponible. Hilo de la interfaz."""
        if not self.corriendo or sesion != self.sesion:
            return

        if self.senal_perdida:
            motivo = self.motivo_corte or "Se corto la senal de la camara"
            self.detener()
            self._set_estado(self._consejo_para(motivo), COLOR["alerta"])
            return

        with self.candado_salida:
            paquete = self.ultima_salida
            self.ultima_salida = None

        # Sin frame nuevo no se repinta: reprocesar el mismo no aporta
        # nada y solo gasta el hilo de la interfaz.
        if paquete is None:
            self.after(5, lambda: self._pintar(sesion))
            return

        salida, medidas = paquete

        if not self.midiendo:
            self.midiendo = True
            self._set_estado("Midiendo")

        self._mostrar_medidas(medidas)
        self._mostrar_frame(salida)

        self.after(MS_ENTRE_PINTADOS, lambda: self._pintar(sesion))

    def _mostrar_medidas(self, medidas):
        hay_alerta = self.alerta_previa

        if "rodilla" in medidas:
            self.valores["rodilla"].configure(
                text=f"{medidas['rodilla']:.0f}°"
            )
            self.valores["cadera"].configure(
                text=f"{medidas['cadera']:.0f}°"
            )

        if "joroba" in medidas:
            postura, hay_alerta = clasificar_postura(
                medidas["inclinacion"], medidas["joroba"], self.alerta_previa
            )
            self.alerta_previa = hay_alerta

            self.valores["inclinacion"].configure(
                text=f"{medidas['inclinacion']:.0f}°"
            )
            self.valores["joroba"].configure(text=f"{medidas['joroba']:.2f}")

            # El texto ya distingue los casos por si solo; el campo de
            # color es lo que se ve a tres metros.
            self._set_veredicto(
                postura,
                COLOR["alerta"] if hay_alerta else COLOR["correcto"],
                COLOR["sobre_campo"],
            )

        self._contar_repeticion(medidas.get("rodilla"), hay_alerta)

    # ---------------------------------
    # Perfil corporal
    # ---------------------------------

    def _medir_perfil(self):
        """Toma las proporciones de la persona, de pie y quieta.

        Hace falta la medicion en marcha: lo que se mide son los
        landmarks que ya esta produciendo el hilo de proceso.
        """

        if not self.corriendo:
            self._set_estado(
                "Empieza la medicion antes de medir el perfil.",
                COLOR["alerta"],
            )
            return

        texto = self.entry_estatura.get().strip()

        if not texto.replace(".", "", 1).isdigit() or not 100 <= float(texto) <= 230:
            self._set_estado(
                "Escribe tu estatura en centimetros (entre 100 y 230).",
                COLOR["alerta"],
            )
            return

        self.estatura_pedida = float(texto)
        self.capturas_perfil = []
        self.boton_perfil.configure(state="disabled")
        self._set_estado("De pie y quieto, de perfil a la camara...")
        self._revisar_perfil()

    def _revisar_perfil(self):
        """Espera a que el hilo junte suficientes capturas."""

        capturas = self.capturas_perfil

        if capturas is None or not self.corriendo:
            self.boton_perfil.configure(state="normal")
            return

        if len(capturas) < CAPTURAS_PARA_PERFIL:
            self._set_estado(
                f"Midiendo perfil... {len(capturas)}/{CAPTURAS_PARA_PERFIL}"
            )
            self.after(100, self._revisar_perfil)
            return

        self.capturas_perfil = None
        self.boton_perfil.configure(state="normal")

        try:
            self.perfil = PerfilCorporal.desde_muestras(
                self.estatura_pedida, capturas
            )
            self.perfil.guardar()
        except (ValueError, ZeroDivisionError) as e:
            self._set_estado(f"No se pudo medir el perfil: {e}", COLOR["alerta"])
            return

        tronco = self.perfil.en_cm("tronco")
        femur = self.perfil.en_cm("femur")

        self._set_estado(
            f"Perfil guardado: tronco {tronco:.0f} cm, muslo {femur:.0f} cm",
            COLOR["correcto"],
        )

    def _contar_repeticion(self, angulo_rodilla, hay_alerta):
        """Avanza la maquina de estados y avisa al cerrar una serie."""

        terminada = self.contador.actualizar(angulo_rodilla, hay_alerta)

        self.valores["repeticiones"].configure(
            text=str(self.contador.repeticiones)
        )

        if terminada is None:
            return

        # El veredicto de la repeticion pisa al de postura un momento:
        # al terminar de subir es cuando se mira, y es lo que importa.
        self._set_veredicto(
            terminada.resumen(),
            COLOR["alerta"] if not terminada.profunda else COLOR["correcto"],
            COLOR["sobre_campo"],
        )

        self._set_estado(
            f"Repeticion {terminada.numero}: "
            f"minimo {terminada.angulo_minimo:.0f}°"
        )

        self.valores["fps"].configure(text=f"{self.fps_medido:.0f} fps")

    def _mostrar_frame(self, output):
        alto, ancho = output.shape[:2]
        caja_ancho, caja_alto = self._video_box

        if caja_ancho > 10 and caja_alto > 10 and alto > 0 and ancho > 0:
            # Sin pasar de ANCHO_MAX_VIDEO: ampliar mas no anade
            # detalle (el frame no lo tiene) y el coste de subirlo a Tk
            # crece con el area. A 1000 px de ancho eran 15 ms por
            # frame; a 960 con paste son unos 8.
            escala = min(caja_ancho / ancho, caja_alto / alto,
                         ANCHO_MAX_VIDEO / ancho)
            nuevo_ancho = max(1, int(ancho * escala))
            nuevo_alto = max(1, int(alto * escala))

            if (nuevo_ancho, nuevo_alto) != (ancho, alto):
                output = cv2.resize(output, (nuevo_ancho, nuevo_alto))

        imagen_pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

        # Reutilizar el PhotoImage con paste() en vez de crear uno
        # nuevo cada frame: crearlo costaba 15 ms, pegarlo cuesta 8.
        # Solo se recrea cuando cambia el tamano.
        if (self.imagen_tk is None
                or self.imagen_tk.width() != imagen_pil.width
                or self.imagen_tk.height() != imagen_pil.height):
            self.imagen_tk = ImageTk.PhotoImage(image=imagen_pil)
            self.label_video.configure(image=self.imagen_tk)
        else:
            self.imagen_tk.paste(imagen_pil)

        self.label_video.image = self.imagen_tk


# =================================
# Ejecutar aplicacion
# =================================

if __name__ == "__main__":
    app = PoseApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
