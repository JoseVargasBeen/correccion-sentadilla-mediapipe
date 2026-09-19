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

USE_SEGMENTATION = True
MASK_THRESH = 0.2

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
# Escala azul sobre base oscura. Se centralizan aqui en vez de repartir
# hex sueltos por la interfaz, para que el cambio de color se haga en un
# solo sitio y para que la version Flutter (Fase 5) parta de los mismos
# valores.
#
# Verde y rojo se quedan fuera de la escala azul a proposito: son los
# unicos dos colores que significan algo (postura correcta / alerta) y
# tienen que distinguirse del resto de la interfaz.

COLOR = {
    "fondo": "#0F172A",
    "superficie": "#1B2336",
    "superficie_alta": "#272F42",
    "borde": "#475569",
    "texto": "#F8FAFC",
    "texto_tenue": "#94A3B8",
    "primario": "#2563EB",
    "primario_hover": "#1D4ED8",
    "sobre_primario": "#FFFFFF",
    "dato": "#60A5FA",
    "exito": "#22C55E",
    "peligro": "#EF4444",
    "peligro_hover": "#DC2626",
    "sobre_peligro": "#FFFFFF",
}

ESPACIO = {"xs": 4, "sm": 8, "md": 12, "lg": 20}

# Los mismos colores para lo que se dibuja sobre el video. OpenCV usa
# BGR, no RGB, asi que van invertidos respecto al hex de arriba.
BGR_SKELETON = (250, 165, 96)   # #60A5FA
BGR_TRONCO = (235, 99, 37)      # #2563EB

# Barlow es la tipografia elegida para el proyecto, pero no viene
# instalada en Windows. Si no esta, se cae a la fuente del sistema en
# vez de dejar que Tk elija una cualquiera en silencio.
FAMILIAS_TITULO = ("Barlow Condensed", "Barlow", "Segoe UI Semibold", "Segoe UI")
FAMILIAS_TEXTO = ("Barlow", "Segoe UI", "Arial")


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
# Umbrales (placeholder, pendientes de calibrar con datos reales)
# =================================

UMBRAL_INCLINACION = 20.0
UMBRAL_JOROBA = 0.20


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


def clasificar_postura(angulo_inclinacion, indice_joroba):
    """Devuelve (texto, hay_alerta).

    El booleano se usa para resaltar la postura en la interfaz; el
    texto por si solo ya dice que pasa, el color solo lo refuerza.
    """
    etiquetas = []

    if abs(angulo_inclinacion) > UMBRAL_INCLINACION:
        lado = "adelante" if angulo_inclinacion > 0 else "atras"
        etiquetas.append(f"Inclinado {lado}")

    if abs(indice_joroba) > UMBRAL_JOROBA:
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
        self.after(0, lambda: self.state("zoomed"))
        self.bind("<F11>", self._alternar_pantalla_completa)
        self.bind("<Escape>", self._salir_pantalla_completa)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.configure(fg_color=COLOR["fondo"])

        self.cap = None
        self.detector = None
        self.corriendo = False

        self.start_time = 0.0
        self.previous_time = 0.0

        self._construir_fuentes()
        self._construir_ui()

    def _construir_fuentes(self):
        # CTkFont necesita que la ventana raiz ya exista, por eso se
        # crean aqui y no como constantes de modulo.
        titulo = primera_familia_disponible(FAMILIAS_TITULO)
        texto = primera_familia_disponible(FAMILIAS_TEXTO)

        self.fuente_seccion = ctk.CTkFont(family=titulo, size=15, weight="bold")
        self.fuente_texto = ctk.CTkFont(family=texto, size=14)
        self.fuente_valor = ctk.CTkFont(family=texto, size=14, weight="bold")
        self.fuente_postura = ctk.CTkFont(family=titulo, size=22, weight="bold")
        self.fuente_estado = ctk.CTkFont(family=texto, size=12)

        # Los numeros son lo que se mira de reojo mientras se hace el
        # ejercicio: van grandes y con su unidad aparte, mas pequena,
        # para que la cifra no compita con el "grados".
        self.fuente_dato = ctk.CTkFont(family=titulo, size=30, weight="bold")
        self.fuente_unidad = ctk.CTkFont(family=texto, size=12)
        self.fuente_etiqueta = ctk.CTkFont(family=texto, size=11)

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

        # Panel fijo a la derecha (evita que sus controles se
        # deformen al maximizar); el video ocupa todo el resto.
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0, minsize=400)
        self.grid_rowconfigure(0, weight=1)

        # --- Panel de video (izquierda) ---
        self.frame_video = ctk.CTkFrame(self, fg_color="black")
        self.frame_video.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.frame_video.grid_columnconfigure(0, weight=1)
        self.frame_video.grid_rowconfigure(0, weight=1)

        self.label_video = ctk.CTkLabel(self.frame_video, text="")
        self.label_video.grid(row=0, column=0, sticky="nsew")

        # Tamano disponible para el video, actualizado cuando el
        # usuario redimensiona la ventana.
        self._video_box = (800, 600)
        self.frame_video.bind("<Configure>", self._on_video_resize)

        # --- Panel lateral (derecha) ---
        # El contenedor fija el ancho y el estado se ancla abajo.
        #
        # Aqui hubo un CTkScrollableFrame y se quito: su canvas interno
        # no repinta lo anterior cuando el contenido cambia de altura
        # (la etiqueta de postura crece y encoge segun el texto) y el
        # bucle de video no deja hueco al redibujado, asi que los
        # widgets quedaban superpuestos unos sobre otros. Con la
        # ventana maximizada todo cabe sin scroll.
        lateral = ctk.CTkFrame(self, width=400, fg_color=COLOR["superficie"])
        lateral.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        lateral.grid_propagate(False)
        lateral.grid_columnconfigure(0, weight=1)
        lateral.grid_rowconfigure(0, weight=1)

        panel = ctk.CTkFrame(lateral, fg_color="transparent")
        panel.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(
            panel, text="Fuente de camara", font=self.fuente_seccion,
            text_color=COLOR["texto"],
        ).pack(pady=(ESPACIO["sm"], 0), anchor="w", padx=ESPACIO["md"])

        self.tipo_camara = ctk.CTkSegmentedButton(
            panel, values=["PC (webcam)", "Celular (WiFi)"],
            command=self._cambiar_tipo_camara,
            font=self.fuente_texto,
            selected_color=COLOR["primario"],
            selected_hover_color=COLOR["primario_hover"],
            text_color=COLOR["texto"],
        )
        self.tipo_camara.set("PC (webcam)")
        self.tipo_camara.pack(pady=ESPACIO["sm"], padx=ESPACIO["md"], fill="x")

        # Contenedor fijo para los campos de la fuente. Sin el, al
        # cambiar de opcion el pack() reinsertaba el frame al final del
        # panel y los campos saltaban debajo de todo.
        self.frame_fuente = ctk.CTkFrame(panel, fg_color="transparent")
        self.frame_fuente.pack(
            pady=ESPACIO["sm"], padx=ESPACIO["md"], fill="x"
        )

        # --- Opcion: webcam del PC ---
        self.frame_fuente_pc = ctk.CTkFrame(
            self.frame_fuente, fg_color="transparent"
        )

        # Etiqueta visible: el placeholder desaparece al escribir y
        # deja el campo sin identificar.
        ctk.CTkLabel(
            self.frame_fuente_pc, text="Indice de camara",
            font=self.fuente_texto, text_color=COLOR["texto_tenue"],
        ).pack(anchor="w")

        self.entry_indice_pc = ctk.CTkEntry(
            self.frame_fuente_pc, placeholder_text="0, 1, ...",
            font=self.fuente_texto, border_color=COLOR["borde"],
        )
        self.entry_indice_pc.insert(0, "0")
        self.entry_indice_pc.pack(fill="x", pady=(ESPACIO["xs"], 0))

        # --- Opcion: celular por IP Webcam ---
        # Solo se escribe el ultimo numero de la IP (el que suele
        # cambiar); el prefijo y el puerto quedan fijos.
        self.frame_fuente_celular = ctk.CTkFrame(
            self.frame_fuente, fg_color="transparent"
        )

        ctk.CTkLabel(
            self.frame_fuente_celular, text="Direccion del celular",
            font=self.fuente_texto, text_color=COLOR["texto_tenue"],
        ).pack(anchor="w")

        fila_ip = ctk.CTkFrame(self.frame_fuente_celular, fg_color="transparent")
        fila_ip.pack(fill="x", pady=(ESPACIO["xs"], 0))

        ctk.CTkLabel(
            fila_ip, text=f"http://{PREFIJO_IP_CELULAR}",
            font=self.fuente_texto, text_color=COLOR["texto"],
        ).pack(side="left")

        self.entry_ip_final = ctk.CTkEntry(
            fila_ip, width=56, font=self.fuente_texto,
            border_color=COLOR["borde"],
        )
        self.entry_ip_final.insert(0, "8")
        self.entry_ip_final.pack(side="left", padx=ESPACIO["xs"])

        ctk.CTkLabel(
            fila_ip, text=f":{PUERTO_IP_WEBCAM}/video",
            font=self.fuente_texto, text_color=COLOR["texto"],
        ).pack(side="left")

        self.frame_fuente_pc.pack(fill="x")

        ctk.CTkLabel(
            panel, text="Rotacion", font=self.fuente_seccion,
            text_color=COLOR["texto"],
        ).pack(pady=(ESPACIO["md"], 0), anchor="w", padx=ESPACIO["md"])

        self.combo_rotacion = ctk.CTkOptionMenu(
            panel, values=list(ROTACIONES.keys()), font=self.fuente_texto,
            fg_color=COLOR["borde"], button_color=COLOR["primario"],
            button_hover_color=COLOR["primario_hover"],
            text_color=COLOR["texto"],
        )
        self.combo_rotacion.pack(pady=ESPACIO["sm"], padx=ESPACIO["md"], fill="x")

        botones = ctk.CTkFrame(panel, fg_color="transparent")
        botones.pack(pady=ESPACIO["md"], padx=ESPACIO["md"], fill="x")

        # height=40 deja los botones sobre el minimo comodo de click y
        # los iguala entre si.
        self.boton_iniciar = ctk.CTkButton(
            botones, text="Iniciar", command=self.iniciar, height=40,
            font=self.fuente_valor,
            fg_color=COLOR["primario"], hover_color=COLOR["primario_hover"],
            text_color=COLOR["sobre_primario"],
        )
        self.boton_iniciar.pack(
            side="left", expand=True, fill="x", padx=(0, ESPACIO["xs"])
        )

        self.boton_detener = ctk.CTkButton(
            botones, text="Detener", command=self.detener, height=40,
            font=self.fuente_valor,
            fg_color=COLOR["peligro"], hover_color=COLOR["peligro_hover"],
            text_color=COLOR["sobre_peligro"],
        )
        self.boton_detener.pack(
            side="left", expand=True, fill="x", padx=(ESPACIO["xs"], 0)
        )

        # --- Postura: es la salida principal del proyecto, va primero
        # y con mas peso visual que las metricas de apoyo. ---
        tarjeta_postura = ctk.CTkFrame(
            panel, fg_color=COLOR["fondo"], border_width=2,
            border_color=COLOR["borde"],
        )
        tarjeta_postura.pack(
            pady=(ESPACIO["sm"], ESPACIO["md"]), padx=ESPACIO["md"], fill="x"
        )

        ctk.CTkLabel(
            tarjeta_postura, text="POSTURA", font=self.fuente_estado,
            text_color=COLOR["texto_tenue"],
        ).pack(pady=(ESPACIO["sm"], 0), padx=ESPACIO["md"], anchor="w")

        # height fijo: el texto ocupa una o dos lineas segun la postura
        # detectada. Sin reservar el alto, la tarjeta crece y encoge
        # varias veces por segundo y arrastra todo el panel con ella.
        self.label_postura = ctk.CTkLabel(
            tarjeta_postura, text="--", font=self.fuente_postura,
            text_color=COLOR["texto_tenue"], wraplength=330, justify="left",
            height=68, anchor="w",
        )
        self.label_postura.pack(
            pady=(0, ESPACIO["sm"]), padx=ESPACIO["md"], anchor="w"
        )

        ctk.CTkLabel(
            panel, text="Valores en vivo", font=self.fuente_seccion,
            text_color=COLOR["texto"],
        ).pack(pady=(ESPACIO["sm"], ESPACIO["xs"]), anchor="w", padx=ESPACIO["md"])

        # Cuadricula de dos columnas: los cuatro angulos se leen de un
        # vistazo mientras se hace el ejercicio, sin buscar el numero al
        # final de una linea de texto.
        self.valores = {}

        rejilla = ctk.CTkFrame(panel, fg_color="transparent")
        rejilla.pack(padx=ESPACIO["md"], fill="x")
        rejilla.grid_columnconfigure(0, weight=1, uniform="dato")
        rejilla.grid_columnconfigure(1, weight=1, uniform="dato")

        metricas = [
            ("rodilla", "RODILLA", "grados"),
            ("cadera", "CADERA", "grados"),
            ("inclinacion", "INCLINACION", "grados"),
            ("joroba", "ENCORVAMIENTO", "indice"),
        ]

        for posicion, (clave, etiqueta, unidad) in enumerate(metricas):
            fila, columna = divmod(posicion, 2)

            tarjeta = ctk.CTkFrame(rejilla, fg_color=COLOR["superficie_alta"])
            tarjeta.grid(
                row=fila, column=columna, padx=ESPACIO["xs"],
                pady=ESPACIO["xs"], sticky="nsew",
            )

            ctk.CTkLabel(
                tarjeta, text=etiqueta, font=self.fuente_etiqueta,
                text_color=COLOR["texto_tenue"],
            ).pack(pady=(ESPACIO["sm"], 0), padx=ESPACIO["sm"])

            valor_label = ctk.CTkLabel(
                tarjeta, text="--", font=self.fuente_dato,
                text_color=COLOR["dato"],
            )
            valor_label.pack(padx=ESPACIO["sm"])

            ctk.CTkLabel(
                tarjeta, text=unidad, font=self.fuente_unidad,
                text_color=COLOR["texto_tenue"],
            ).pack(pady=(0, ESPACIO["sm"]), padx=ESPACIO["sm"])

            self.valores[clave] = valor_label

        # Los FPS son diagnostico, no una medida del ejercicio: van
        # aparte y sin el peso visual de los angulos.
        fila_fps = ctk.CTkFrame(panel, fg_color="transparent")
        fila_fps.pack(pady=(ESPACIO["sm"], 0), padx=ESPACIO["lg"], fill="x")

        ctk.CTkLabel(
            fila_fps, text="FPS", font=self.fuente_texto,
            text_color=COLOR["texto_tenue"],
        ).pack(side="left")

        self.valores["fps"] = ctk.CTkLabel(
            fila_fps, text="--", font=self.fuente_valor,
            text_color=COLOR["texto"],
        )
        self.valores["fps"].pack(side="right")

        # Anclado al contenedor, no al area con scroll: el estado debe
        # verse siempre.
        self.label_estado = ctk.CTkLabel(
            lateral, text="Detenido", font=self.fuente_estado,
            text_color=COLOR["texto_tenue"],
        )
        self.label_estado.grid(
            row=1, column=0, pady=ESPACIO["sm"], padx=ESPACIO["md"]
        )

        self._actualizar_botones()

    # ---------------------------------
    # Control de camara / deteccion
    # ---------------------------------

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

    def _set_estado(self, texto, color):
        self.label_estado.configure(text=texto, text_color=color)

    def _cambiar_tipo_camara(self, seleccion):

        if seleccion == "PC (webcam)":
            self.frame_fuente_celular.pack_forget()
            self.frame_fuente_pc.pack(fill="x")
        else:
            self.frame_fuente_pc.pack_forget()
            self.frame_fuente_celular.pack(fill="x")

    def _on_video_resize(self, event):
        self._video_box = (event.width, event.height)

    def _obtener_fuente(self):

        if self.tipo_camara.get() == "PC (webcam)":
            texto = self.entry_indice_pc.get().strip()
            return int(texto) if texto.isdigit() else 0

        ultimo_numero = self.entry_ip_final.get().strip()
        return f"http://{PREFIJO_IP_CELULAR}{ultimo_numero}:{PUERTO_IP_WEBCAM}/video"

    def iniciar(self):

        if self.corriendo:
            return

        fuente = self._obtener_fuente()

        self.cap = LectorCamara(fuente)

        if not self.cap.abierto():
            self._set_estado(f"No se pudo abrir: {fuente}", COLOR["peligro"])
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
            output_segmentation_masks=USE_SEGMENTATION,
        )

        self.detector = vision.PoseLandmarker.create_from_options(options)

        self.start_time = time.time()
        self.previous_time = time.time()

        self.corriendo = True
        self._set_estado("Corriendo", COLOR["exito"])
        self._actualizar_botones()

        self._actualizar_frame()

    def detener(self):

        self.corriendo = False

        if self.cap is not None:
            self.cap.detener()
            self.cap = None

        if self.detector is not None:
            self.detector.close()
            self.detector = None

        self._reiniciar_valores()
        self._set_estado("Detenido", COLOR["texto_tenue"])
        self._actualizar_botones()

    def _reiniciar_valores(self):
        """Al detener, los ultimos numeros dejan de ser ciertos."""
        for valor_label in self.valores.values():
            valor_label.configure(text="--")

        self.label_postura.configure(
            text="--", text_color=COLOR["texto_tenue"]
        )

    def on_closing(self):
        self.detener()
        self.destroy()

    # ---------------------------------
    # Bucle principal (dirigido por eventos, no bloqueante)
    # ---------------------------------

    def _actualizar_frame(self):

        if not self.corriendo:
            return

        if not self.cap.corriendo:
            # El mensaje va despues de detener(), que reescribe el
            # estado a "Detenido" y si no borraria el aviso al instante.
            self.detener()
            self._set_estado("Se perdio la senal de camara", COLOR["peligro"])
            return

        ok, frame_bgr = self.cap.leer()

        # Todavia no hay frame nuevo: no es un error, solo se reintenta
        # en el proximo ciclo sin procesar nada.
        if not ok:
            self.after(5, self._actualizar_frame)
            return

        rotacion = ROTACIONES[self.combo_rotacion.get()]

        if rotacion is not None:
            frame_bgr = cv2.rotate(frame_bgr, rotacion)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp_ms = int((time.time() - self.start_time) * 1000)

        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        output = frame_bgr.copy()

        # --- Segmentacion ---

        if USE_SEGMENTATION and result.segmentation_masks:

            mask = np.squeeze(result.segmentation_masks[0].numpy_view())
            mask3 = np.repeat(mask[:, :, None], 3, axis=2)

            fondo_gris = cv2.cvtColor(
                cv2.cvtColor(output, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR
            )

            output = np.where(mask3 > MASK_THRESH, output, fondo_gris)

        # --- Landmarks + calculo de variables (sin dibujar texto) ---

        if result.pose_landmarks:

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

                angulo_rodilla = calcular_angulo(
                    points[CADERA_D], points[RODILLA_D], points[TOBILLO_D]
                )
                angulo_cadera = calcular_angulo(
                    points[HOMBRO_D], points[CADERA_D], points[RODILLA_D]
                )

                # La unidad va en su propia etiqueta dentro de la
                # tarjeta, aqui solo la cifra.
                self.valores["rodilla"].configure(text=f"{angulo_rodilla:.0f}")
                self.valores["cadera"].configure(text=f"{angulo_cadera:.0f}")

            necesarios_espalda = [HOMBRO_I, HOMBRO_D, CADERA_I, CADERA_D, OREJA_D]

            if all(points[i] is not None for i in necesarios_espalda):

                hombro_medio = punto_medio(points[HOMBRO_I], points[HOMBRO_D])
                cadera_medio = punto_medio(points[CADERA_I], points[CADERA_D])

                angulo_inclinacion = angulo_respecto_vertical(
                    cadera_medio, hombro_medio
                )
                indice_joroba = indice_encorvamiento(
                    points[OREJA_D], hombro_medio, cadera_medio
                )

                postura, hay_alerta = clasificar_postura(
                    angulo_inclinacion, indice_joroba
                )

                self.valores["inclinacion"].configure(
                    text=f"{angulo_inclinacion:.1f}"
                )
                self.valores["joroba"].configure(text=f"{indice_joroba:.2f}")

                # El texto ya distingue los casos; el color solo lo
                # refuerza, no es el unico canal.
                self.label_postura.configure(
                    text=postura,
                    text_color=(
                        COLOR["peligro"] if hay_alerta else COLOR["exito"]
                    ),
                )

                cv2.line(
                    output,
                    (int(cadera_medio[0]), int(cadera_medio[1])),
                    (int(hombro_medio[0]), int(hombro_medio[1])),
                    BGR_TRONCO, grosor_linea,
                )

        # --- FPS ---

        current_time = time.time()
        fps = 1.0 / max(1e-6, current_time - self.previous_time)
        self.previous_time = current_time

        self.valores["fps"].configure(text=f"{fps:.1f}")

        # --- Mostrar frame en la interfaz ---
        # Escalar para ocupar el espacio disponible del panel de
        # video (sea vertical u horizontal), sin deformar la imagen.

        alto, ancho = output.shape[:2]
        caja_ancho, caja_alto = self._video_box

        if caja_ancho > 10 and caja_alto > 10 and alto > 0 and ancho > 0:
            escala = min(caja_ancho / ancho, caja_alto / alto)
            nuevo_ancho = max(1, int(ancho * escala))
            nuevo_alto = max(1, int(alto * escala))
            output = cv2.resize(output, (nuevo_ancho, nuevo_alto))

        imagen_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        imagen_pil = Image.fromarray(imagen_rgb)
        imagen_tk = ImageTk.PhotoImage(image=imagen_pil)

        self.label_video.configure(image=imagen_tk)
        self.label_video.image = imagen_tk  # evitar garbage collection

        self.after(15, self._actualizar_frame)


# =================================
# Ejecutar aplicacion
# =================================

if __name__ == "__main__":
    app = PoseApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
