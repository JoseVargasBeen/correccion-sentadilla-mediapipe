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
from PIL import Image, ImageTk

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

ROTACIONES = {
    "Sin rotar": None,
    "90 horario": cv2.ROTATE_90_CLOCKWISE,
    "90 antihorario": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "180 grados": cv2.ROTATE_180,
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
# Sigue siendo una estimacion sobre proporciones medias, no una
# calibracion. Para ajustarlo al cuerpo y al angulo de camara reales,
# usar calibrar_encorvamiento.py.
UMBRAL_INCLINACION = 20.0
UMBRAL_JOROBA = 0.10

# Histeresis: al bajar el umbral, los valores rondando el limite
# encienden y apagan la alerta varias veces por segundo y el campo de
# color parpadea. Con margen, encender cuesta mas que apagar.
MARGEN_JOROBA = 0.02
MARGEN_INCLINACION = 3.0


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
        self.captura = cv2.VideoCapture(fuente)

        # Algunos backends respetan esto y ya limitan la cola interna.
        try:
            self.captura.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.frame = None
        self.candado = threading.Lock()
        self.corriendo = False
        self.hilo = None

    def abierto(self):
        return self.captura.isOpened()

    def iniciar(self):
        self.corriendo = True
        self.hilo = threading.Thread(target=self._bucle, daemon=True)
        self.hilo.start()

    def _bucle(self):
        while self.corriendo:
            ok, frame = self.captura.read()

            if not ok:
                self.corriendo = False
                break

            # Sobrescribe el anterior: si no alcanzo a procesarse, se
            # descarta a proposito.
            with self.candado:
                self.frame = frame

    def leer(self):
        with self.candado:
            frame = self.frame
            self.frame = None

        return (frame is not None), frame

    def detener(self):
        self.corriendo = False

        if self.hilo is not None:
            self.hilo.join(timeout=1.0)
            self.hilo = None

        self.captura.release()


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
    >>> clasificar_postura(0.0, 0.09, alerta_previa=True)[1]
    True
    >>> clasificar_postura(0.0, 0.09, alerta_previa=False)[1]
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

        # El hilo de proceso no puede consultar el desplegable: Tk solo
        # admite llamadas desde el hilo principal. La interfaz deja
        # aqui el valor elegido y el hilo lo lee.
        self.rotacion_actual = "Sin rotar"

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

        self.frame_fuente_pc.pack(fill="x")

        self._etiqueta_control(controles, 2, "Giro")
        self.combo_rotacion = ctk.CTkOptionMenu(
            controles, values=list(ROTACIONES.keys()),
            font=self.fuente_texto, corner_radius=0,
            fg_color=COLOR["campo"], button_color=COLOR["filete"],
            button_hover_color=COLOR["tenue"],
            text_color=COLOR["texto"],
        )
        self.combo_rotacion.grid(row=2, column=1, sticky="ew",
                                 pady=ESPACIO["xs"])

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

    def _cambiar_rotacion(self, seleccion):
        self.rotacion_actual = seleccion

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
            return f"http://{PREFIJO_IP_CELULAR}{ultimo}:{PUERTO_IP_WEBCAM}/video"

        return completar_direccion(self.entry_ip_completa.get())

    def iniciar(self):

        if self.corriendo:
            return

        fuente = self._obtener_fuente()

        if fuente == "":
            self._set_estado(
                "Escribe la direccion del celular antes de empezar.",
                COLOR["alerta"],
            )
            return

        self.cap = LectorCamara(fuente)

        if not self.cap.abierto():
            # El error dice que hacer, no solo que fallo.
            if self.tipo_camara.get() == "Esta PC":
                aviso = f"No hay camara en el indice {fuente}. Prueba con 1."
            else:
                aviso = "El celular no responde. Revisa que IP Webcam este abierto y en la misma red."

            self._set_estado(aviso, COLOR["alerta"])
            self.cap = None
            return

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
        self.ultima_salida = None
        self.rotacion_actual = self.combo_rotacion.get()

        self.corriendo = True
        self.hilo_proceso = threading.Thread(
            target=self._bucle_proceso, daemon=True
        )
        self.hilo_proceso.start()

        self._set_estado("Midiendo")
        self._set_veredicto(
            "Buscando", COLOR["filete"], COLOR["tenue"]
        )
        self._actualizar_botones()

        self._pintar()

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

    def _reiniciar_valores(self):
        """Al detener, los ultimos numeros dejan de ser ciertos."""
        for clave, valor_label in self.valores.items():
            valor_label.configure(text="" if clave == "fps" else "—")

        self._set_veredicto(
            "Sin lectura", COLOR["filete"], COLOR["tenue"]
        )

    def on_closing(self):
        self.detener()
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
        while self.corriendo:
            ok, frame_bgr = self.cap.leer()

            if not self.cap.corriendo:
                self.senal_perdida = True
                return

            if not ok:
                time.sleep(0.002)
                continue

            rotacion = ROTACIONES[self.rotacion_actual]

            if rotacion is not None:
                frame_bgr = cv2.rotate(frame_bgr, rotacion)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=frame_rgb)
            timestamp_ms = int((time.time() - self.start_time) * 1000)

            try:
                result = self.detector.detect_for_video(mp_image, timestamp_ms)
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

    def _pintar(self):
        """Pinta el ultimo resultado disponible. Hilo de la interfaz."""
        if not self.corriendo:
            return

        if self.senal_perdida:
            self.detener()
            self._set_estado("Se corto la senal de la camara", COLOR["alerta"])
            return

        with self.candado_salida:
            paquete = self.ultima_salida
            self.ultima_salida = None

        # Sin frame nuevo no se repinta: reprocesar el mismo no aporta
        # nada y solo gasta el hilo de la interfaz.
        if paquete is None:
            self.after(5, self._pintar)
            return

        salida, medidas = paquete

        self._mostrar_medidas(medidas)
        self._mostrar_frame(salida)

        self.after(MS_ENTRE_PINTADOS, self._pintar)

    def _mostrar_medidas(self, medidas):
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
