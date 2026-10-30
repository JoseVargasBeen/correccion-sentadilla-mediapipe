package com.josed.camcel

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.net.wifi.WifiManager
import android.os.Build
import android.view.WindowManager
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors

/**
 * Puente entre Dart y codigo nativo de Android (method channel).
 *
 * Convertir YUV a JPEG en Dart es lento: recorre pixel por pixel en un
 * solo nucleo. Android trae YuvImage.compressToJpeg(), que hace lo mismo
 * en codigo nativo optimizado. Esta clase lo expone a Dart.
 *
 * La compresion corre en un hilo aparte para no trabar la UI de Android
 * (el preview de la camara), y la respuesta se devuelve en el hilo
 * principal, que es donde MethodChannel.Result exige ser llamado.
 */
class MainActivity : FlutterActivity() {

    private val nombreCanal = "camcel/jpeg"
    private val ejecutor = Executors.newSingleThreadExecutor()

    // Mientras se guarde esta reserva la red sigue viva; al cerrarla,
    // Android la apaga.
    private var reservaPuntoDeAcceso: WifiManager.LocalOnlyHotspotReservation? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, nombreCanal)
            .setMethodCallHandler { llamada, resultado ->
                when (llamada.method) {
                    "nv21AJpeg" -> manejarConversion(llamada, resultado)
                    "mantenerPantallaEncendida" -> {
                        mantenerPantallaEncendida(
                            llamada.argument<Boolean>("activar") ?: false
                        )
                        resultado.success(null)
                    }
                    "abrirPuntoDeAcceso" -> abrirPuntoDeAcceso(resultado)
                    "cerrarPuntoDeAcceso" -> {
                        cerrarPuntoDeAcceso()
                        resultado.success(null)
                    }
                    else -> resultado.notImplemented()
                }
            }
    }

    /**
     * FLAG_KEEP_SCREEN_ON evita que la pantalla se apague mientras la
     * app esta en primer plano. Hace falta porque al apagarse la
     * pantalla Android suspende la camara y se corta la transmision.
     *
     * No requiere permisos y se libera solo al salir de la app, a
     * diferencia de un WakeLock, que hay que soltar a mano.
     */
    private fun mantenerPantallaEncendida(activar: Boolean) {
        runOnUiThread {
            if (activar) {
                window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            } else {
                window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            }
        }
    }

    /**
     * Punto de acceso "solo local": el celular crea una red WiFi propia
     * a la que el PC se conecta.
     *
     * Hace falta cuando el celular esta en datos moviles. Esa IP (una
     * 10.x del operador) vive detras del NAT de la red celular y no es
     * alcanzable desde fuera, asi que la unica via es que el celular
     * monte la red y el PC entre en ella.
     *
     * Se usa startLocalOnlyHotspot y no el anclaje normal porque es la
     * unica variante que una app corriente puede encender por si misma
     * y de la que puede leer la clave, que es lo que permite meterla en
     * el QR. A cambio no comparte internet con el PC: solo los conecta
     * entre si, que es justo lo que necesita el video.
     */
    private fun abrirPuntoDeAcceso(resultado: MethodChannel.Result) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) {
            resultado.error("NO_SOPORTADO", "Requiere Android 8 o superior", null)
            return
        }

        if (!tienePermisoDeRed()) {
            pedirPermisoDeRed()
            resultado.error(
                "SIN_PERMISO",
                "Falta el permiso para crear la red. Aceptalo y vuelve a intentar.",
                null,
            )
            return
        }

        if (reservaPuntoDeAcceso != null) {
            resultado.error("YA_ABIERTO", "El punto de acceso ya esta encendido", null)
            return
        }

        val wifi = applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager

        try {
            wifi.startLocalOnlyHotspot(
                object : WifiManager.LocalOnlyHotspotCallback() {
                    override fun onStarted(reserva: WifiManager.LocalOnlyHotspotReservation) {
                        reservaPuntoDeAcceso = reserva

                        val credenciales = leerCredenciales(reserva)

                        if (credenciales == null) {
                            resultado.error(
                                "SIN_CREDENCIALES",
                                "La red se creo pero Android no entrego su clave",
                                null,
                            )
                            return
                        }

                        resultado.success(credenciales)
                    }

                    override fun onFailed(motivo: Int) {
                        reservaPuntoDeAcceso = null
                        resultado.error(
                            "FALLO",
                            "Android no pudo crear la red (codigo $motivo). " +
                                "Suele pasar si el anclaje ya esta encendido.",
                            null,
                        )
                    }

                    override fun onStopped() {
                        reservaPuntoDeAcceso = null
                    }
                },
                null,
            )
        } catch (e: Exception) {
            resultado.error("FALLO", e.message, null)
        }
    }

    private fun leerCredenciales(
        reserva: WifiManager.LocalOnlyHotspotReservation,
    ): Map<String, String>? {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val config = reserva.softApConfiguration
            val ssid = config.ssid ?: return null
            val clave = config.passphrase ?: return null
            return mapOf("ssid" to ssid, "clave" to clave)
        }

        @Suppress("DEPRECATION")
        val config = reserva.wifiConfiguration ?: return null

        @Suppress("DEPRECATION")
        val ssid = config.SSID ?: return null

        @Suppress("DEPRECATION")
        val clave = config.preSharedKey ?: return null

        return mapOf("ssid" to ssid, "clave" to clave)
    }

    private fun cerrarPuntoDeAcceso() {
        reservaPuntoDeAcceso?.close()
        reservaPuntoDeAcceso = null
    }

    private fun tienePermisoDeRed(): Boolean {
        val permiso = permisoDeRed()

        return ContextCompat.checkSelfPermission(this, permiso) ==
            PackageManager.PERMISSION_GRANTED
    }

    private fun pedirPermisoDeRed() {
        ActivityCompat.requestPermissions(this, arrayOf(permisoDeRed()), 1)
    }

    /** Android 13 movio este permiso de "ubicacion" a "dispositivos cercanos". */
    private fun permisoDeRed(): String {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            Manifest.permission.NEARBY_WIFI_DEVICES
        } else {
            Manifest.permission.ACCESS_FINE_LOCATION
        }
    }

    private fun manejarConversion(
        llamada: io.flutter.plugin.common.MethodCall,
        resultado: MethodChannel.Result,
    ) {
        val bytes = llamada.argument<ByteArray>("bytes")
        val ancho = llamada.argument<Int>("ancho")
        val alto = llamada.argument<Int>("alto")
        val calidad = llamada.argument<Int>("calidad") ?: 75

        if (bytes == null || ancho == null || alto == null) {
            resultado.error("ARGUMENTOS", "Faltan argumentos", null)
            return
        }

        ejecutor.execute {
            try {
                val imagenYuv = YuvImage(bytes, ImageFormat.NV21, ancho, alto, null)
                val salida = ByteArrayOutputStream()

                imagenYuv.compressToJpeg(Rect(0, 0, ancho, alto), calidad, salida)

                val jpeg = salida.toByteArray()
                runOnUiThread { resultado.success(jpeg) }
            } catch (e: Exception) {
                runOnUiThread { resultado.error("CONVERSION", e.message, null) }
            }
        }
    }

    override fun onDestroy() {
        ejecutor.shutdown()
        cerrarPuntoDeAcceso()
        super.onDestroy()
    }
}
