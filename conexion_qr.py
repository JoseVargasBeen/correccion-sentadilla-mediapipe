"""
conexion_qr.py

Lee el QR que muestra Camcel y deja el PC listo para recibir el video.

El QR trae un JSON con dos formas:

    {"ip": "192.168.1.14", "puerto": 8080}
        Ya estan en la misma red: solo hay que usar esa direccion.

    {"ip": "192.168.49.1", "puerto": 8080, "ssid": "...", "clave": "..."}
        El celular creo su propia red: hay que entrar en ella primero.

La segunda forma es la que resuelve el caso de solo tener datos
moviles. La IP que Android recibe del operador vive detras de su NAT y
no es alcanzable desde el PC, asi que la unica via es que el celular
monte la red y el PC se pase a ella.
"""

import json
import subprocess
import threading
import time
import xml.sax.saxutils as saxutils

import cv2


TIEMPO_MAXIMO_ESCANEO = 40.0
TIEMPO_MAXIMO_CONEXION = 25.0


class ErrorQr(Exception):
    """Fallo con un mensaje pensado para mostrarse tal cual."""


def leer_datos_qr(texto):
    """Valida el contenido del QR y lo devuelve como diccionario."""

    try:
        datos = json.loads(texto)
    except json.JSONDecodeError:
        raise ErrorQr("Ese QR no es de Camcel.")

    if not isinstance(datos, dict) or "ip" not in datos:
        raise ErrorQr("Ese QR no es de Camcel.")

    return datos


def url_de(datos):
    puerto = datos.get("puerto", 8080)
    return f"http://{datos['ip']}:{puerto}/video"


def _ejecutar(comando, timeout=20):
    """Lanza un comando sin abrir ventana de consola.

    Sin CREATE_NO_WINDOW, cada llamada a netsh hace parpadear una
    consola negra encima de la interfaz.
    """

    return subprocess.run(
        comando,
        capture_output=True,
        text=True,
        timeout=timeout,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def red_actual():
    """SSID al que esta conectado el PC ahora mismo, o None."""

    try:
        salida = _ejecutar(["netsh", "wlan", "show", "interfaces"]).stdout
    except Exception:
        return None

    for linea in salida.splitlines():
        partes = linea.split(":", 1)

        # Se compara el nombre exacto para no confundir "SSID" con
        # "BSSID", que aparece en la linea siguiente.
        if len(partes) == 2 and partes[0].strip().upper() == "SSID":
            return partes[1].strip() or None

    return None


def _perfil_wifi(ssid, clave):
    """XML de perfil que espera netsh para una red WPA2 personal."""

    # Las claves que genera Android son alfanumericas, pero el SSID
    # puede traer cualquier cosa y un & sin escapar rompe el XML
    # entero. Se escapan tambien las comillas: no hace falta dentro de
    # un elemento, pero el parser las devuelve igual y asi no hay que
    # razonar sobre ello.
    escapes = {'"': "&quot;", "'": "&apos;"}

    ssid_xml = saxutils.escape(ssid, escapes)
    clave_xml = saxutils.escape(clave, escapes)

    return f"""<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
  <name>{ssid_xml}</name>
  <SSIDConfig>
    <SSID>
      <name>{ssid_xml}</name>
    </SSID>
  </SSIDConfig>
  <connectionType>ESS</connectionType>
  <connectionMode>manual</connectionMode>
  <MSM>
    <security>
      <authEncryption>
        <authentication>WPA2PSK</authentication>
        <encryption>AES</encryption>
        <useOneX>false</useOneX>
      </authEncryption>
      <sharedKey>
        <keyType>passPhrase</keyType>
        <protected>false</protected>
        <keyMaterial>{clave_xml}</keyMaterial>
      </sharedKey>
    </security>
  </MSM>
</WLANProfile>"""


def conectar_a_red(ssid, clave, al_avanzar=None):
    """Pasa el PC a la red del celular. Devuelve el SSID anterior.

    Guardar el SSID previo permite volver a la red de siempre al
    terminar: la red del celular no da internet.
    """

    def avisar(texto):
        if al_avanzar is not None:
            al_avanzar(texto)

    anterior = red_actual()

    if anterior == ssid:
        return anterior

    avisar(f"Entrando a la red {ssid}")

    import os
    import tempfile

    # netsh solo acepta el perfil desde un archivo.
    ruta = None

    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".xml", delete=False, encoding="utf-8"
        ) as archivo:
            archivo.write(_perfil_wifi(ssid, clave))
            ruta = archivo.name

        resultado = _ejecutar(
            ["netsh", "wlan", "add", "profile", f"filename={ruta}", "user=current"]
        )

        if resultado.returncode != 0:
            raise ErrorQr(
                "Windows no acepto el perfil de la red del celular."
            )
    finally:
        if ruta is not None:
            try:
                os.unlink(ruta)
            except OSError:
                pass

    resultado = _ejecutar(["netsh", "wlan", "connect", f"name={ssid}"])

    if resultado.returncode != 0:
        raise ErrorQr(f"No se pudo entrar a la red {ssid}.")

    # netsh vuelve en cuanto lanza la peticion, no cuando ya esta
    # conectado: hay que esperar a que la interfaz lo confirme.
    limite = time.time() + TIEMPO_MAXIMO_CONEXION

    while time.time() < limite:
        if red_actual() == ssid:
            avisar(f"Conectado a {ssid}")
            return anterior

        time.sleep(0.5)

    raise ErrorQr(
        f"El PC no termino de entrar a {ssid}. Revisa que el punto de "
        "acceso siga encendido en el celular."
    )


class EscanerQr:
    """Busca un QR con la webcam del PC, en un hilo aparte.

    Va en su propio hilo por lo mismo que la captura de video: abrir la
    camara y decodificar bloquean, y hacerlo en el hilo de Tk congela
    la ventana.
    """

    def __init__(self, indice_camara=0):
        self.indice_camara = indice_camara

        self.datos = None
        self.error = None
        self.frame = None
        self.terminado = False

        # Esquinas del QR en el ultimo frame, para que la interfaz
        # pueda marcarlo y se vea que si lo esta cogiendo.
        self.esquinas = None

        self._candado = threading.Lock()
        self._cancelado = False
        self._hilo = None

    def iniciar(self):
        self._hilo = threading.Thread(target=self._bucle, daemon=True)
        self._hilo.start()

    def cancelar(self):
        self._cancelado = True

    def ultimo_frame(self):
        """Devuelve (frame, esquinas) del ultimo fotograma leido."""
        with self._candado:
            return self.frame, self.esquinas

    def _bucle(self):
        captura = None

        try:
            captura = cv2.VideoCapture(self.indice_camara)

            if not captura.isOpened():
                self.error = (
                    "No se pudo abrir la camara del PC para leer el QR."
                )
                return

            detector = cv2.QRCodeDetector()
            limite = time.time() + TIEMPO_MAXIMO_ESCANEO

            while not self._cancelado and time.time() < limite:
                ok, frame = captura.read()

                if not ok:
                    # Sin pausa, una camara que deja de responder deja
                    # el hilo girando a tope hasta agotar el tiempo.
                    time.sleep(0.03)
                    continue

                try:
                    texto, puntos, _ = detector.detectAndDecode(frame)
                except cv2.error:
                    # Un frame borroso puede reventar el decodificador:
                    # se descarta y se sigue con el siguiente.
                    texto, puntos = "", None

                with self._candado:
                    self.frame = frame
                    self.esquinas = puntos

                if not texto:
                    continue

                try:
                    self.datos = leer_datos_qr(texto)
                except ErrorQr as e:
                    # Un QR cualquiera (un producto, un pago) no debe
                    # cortar el escaneo: se ignora y se sigue buscando.
                    self.error = str(e)
                    continue

                self.error = None
                return

            if not self._cancelado and self.datos is None:
                self.error = "No se vio ningun QR. Acercalo mas a la camara."

        except Exception as e:
            self.error = f"Error leyendo el QR: {e}"

        finally:
            self.terminado = True

            if captura is not None:
                captura.release()


def volver_a_red(ssid):
    """Regresa a la red anterior. Sin ruido si falla: es cortesia."""

    if not ssid:
        return

    try:
        _ejecutar(["netsh", "wlan", "connect", f"name={ssid}"])
    except Exception:
        pass
