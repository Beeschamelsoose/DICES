import os
import cv2
from picamera2 import Picamera2

# Font- und QT-Warnungen unterdrücken
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

def test_camera_stream():
    picam = Picamera2()

    # Standard-Konfiguration der rpicam-apps für die Vorschau
    config = picam.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam.configure(config)
    picam.start()

    print("\n" + "="*60)
    print("Reiner Kamera-Test aktiv. Drücke 'q' im Videofenster zum Beenden.")
    print("="*60 + "\n")

    # Einmalige Ausgabe der Array-Eigenschaften beim ersten Frame
    first_frame = True

    try:
        while True:
            # Holt das rohe NumPy-Array aus dem Kameraspeicher
            frame_raw = picam.capture_array()

            if first_frame:
                print("--- SENSOR DATA INFO ---")
                print(f"Array-Typ: {type(frame_raw)}")
                print(f"Array-Dimensionen (Shape): {frame_raw.shape}")
                print(f"Datentyp (dtype): {frame_raw.dtype}")
                print("-" * 24 + "\n")
                print("Teste jetzt die Farbvarianten im Code, falls die Anzeige falsch ist.\n")
                first_frame = False

            # =================================================================
            # SCHALTZENTRALE FÜR DEN FARB-TEST:
            # Kommentiere immer nur EINE der folgenden Varianten ein!
            # =================================================================

            # VARIANTE A (Standard RGB nach BGR - die wahrscheinlichste für Picamera2):
           # frame = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2BGR)

            # VARIANTE B (Direkte Ausgabe ohne Konvertierung):
            frame = frame_raw

            # VARIANTE C (Falls der Sensor hartnäckig YUV/YUYV liefert):
            # try:
            #     frame = cv2.cvtColor(frame_raw, cv2.COLOR_YUV2BGR_YUYV)
            # except Exception as e:
            #     frame = frame_raw

            # =================================================================

            # Text-Overlay zur Orientierung im Fenster
            cv2.putText(frame, "Reiner Kamera-Feed - Keine Verarbeitung", (20, 40),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

            # Bild anzeigen
            cv2.imshow("Pure RPi Cam Test", frame)

            # Beenden mit 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        picam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera_stream()
