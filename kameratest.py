from picamera2 import Picamera2
import cv2
import time

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": 'BGR', "size": (1280, 720)}
)
picam2.configure(config)
picam2.start_preview()
picam2.start()

print("Kamerabild läuft! Drücke Ctrl+C zum Stoppen...")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Beendet.")
    
picam2.stop()
