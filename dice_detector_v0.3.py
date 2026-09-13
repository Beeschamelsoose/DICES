import os
import cv2
import numpy as np
from picamera2 import Picamera2
import time

os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"


def create_background_mask(hsv):
    """Erstellt eine Maske fuer dominante Hintergrundfarben mit 3D-HSV-Histogramm."""
    h, s, v = cv2.split(hsv)

    # 3D-Histogramm: H (18), S (16), V (16)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [18, 16, 16], [0, 180, 0, 256, 0, 256])
    hist = hist.astype(np.float32)
    flat = hist.reshape(-1)

    # Dominante Farb-Bins bestimmen (99-Perzentil)
    peak_threshold = np.percentile(flat, 99.0)
    peak_indices = np.where(flat >= peak_threshold)[0]
    if len(peak_indices) < 3:
        peak_threshold = np.percentile(flat, 98.0)
        peak_indices = np.where(flat >= peak_threshold)[0]
    if len(peak_indices) == 0:
        return np.zeros_like(h, dtype=np.uint8)

    # Max. 12 dominante Farbklassen verwenden
    if len(peak_indices) > 12:
        peak_indices = peak_indices[np.argsort(flat[peak_indices])[-12:]]

    # Histogramm-Bin-Grenzen
    h_step, s_step, v_step = 180.0 / 18.0, 256.0 / 16.0, 256.0 / 16.0
    background_mask = np.zeros_like(h, dtype=np.uint8)

    # Fuer jeden dominanten 3D-HSV-Bin einen Bereich markieren
    for idx in peak_indices:
        hi, si, vi = np.unravel_index(int(idx), hist.shape)
        h_center, s_center, v_center = (hi + 0.5) * h_step, (si + 0.5) * s_step, (vi + 0.5) * v_step

        # Toleranz um die Histogramm-Peaks
        dh, ds, dv = 12, 35, 35
        h_low, h_high = h_center - dh, h_center + dh

        # Hue zyklisch behandeln (0-180)
        if h_low < 0:
            hue_mask = (h >= h_low + 180) | (h <= h_high)
        elif h_high >= 180:
            hue_mask = (h >= h_low) | (h <= h_high - 180)
        else:
            hue_mask = (h >= h_low) & (h <= h_high)

        color_mask = hue_mask & (np.abs(s.astype(np.float32) - s_center) <= ds) & (np.abs(v.astype(np.float32) - v_center) <= dv)
        background_mask[color_mask] = 255

    # Morphologische Nachbearbeitung
    bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.morphologyEx(background_mask, cv2.MORPH_CLOSE, bg_kernel)

def detect_individual_dice():
    # Kamera initialisieren
    picam = Picamera2()
    config = picam.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam.configure(config)
    picam.start()

    # Fenster erstellen
    cv2.namedWindow("HSV Hintergrund-Maske")
    cv2.namedWindow("Saettigungs-Wuerfel-Maske")
    cv2.namedWindow("Augen-Maske")
    cv2.namedWindow("RPi rpicam Multi-Dice Detector")

    print("\n" + "=" * 60)
    print("Multi-Dice Detector\nWuerfel: HSV-Histogramm + Saettigung\nAugen: relative Dunkelkeit innerhalb des Wuerfels")
    #print("Druecke 'q' zum Beenden.\n" + "=" * 60 + "\n")

    try:
        while True:
            # Frame mit Kamera erfassen und resize
            frame = picam.capture_array()
            if frame.shape[0] != 480 or frame.shape[1] != 640:
                frame = cv2.resize(frame, (640, 480))
            display_frame = frame.copy()

            # HSV-Umwandlung
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            saturation, gray = hsv[:, :, 1], hsv[:, :, 2]

            # Hintergrund-Maske erzeugen
            background_mask = create_background_mask(hsv)

            # Wuerfel-Kandidaten: niedrige Saettigung = weisse Flaechen
            sat_blurred = cv2.GaussianBlur(saturation, (11, 11), 0)
            _, dice_mask = cv2.threshold(sat_blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            dice_mask[background_mask > 0] = 0  # Hintergrund entfernen

            # Morphologische Bereinigung
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dice_mask = cv2.morphologyEx(dice_mask, cv2.MORPH_CLOSE, kernel)
            dice_mask = cv2.morphologyEx(dice_mask, cv2.MORPH_OPEN, kernel)

            # Raender abschneiden
            h, w = dice_mask.shape
            cv2.rectangle(dice_mask, (0, 0), (w, 15), 0, -1)
            cv2.rectangle(dice_mask, (0, 0), (15, h), 0, -1)
            cv2.rectangle(dice_mask, (w - 15, 0), (w, h), 0, -1)
            cv2.rectangle(dice_mask, (0, h - 15), (w, h), 0, -1)


            # Konturen finden und verarbeiten
            contours, _ = cv2.findContours(dice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            overall_score, dice_results, eyes_debug = 0, [], np.zeros_like(gray)

            last_print = 0

            for cnt in contours:
                # Filterung nach Flaeche und Aspectratio
                area = cv2.contourArea(cnt)
                if not (500 < area < 7500):
                    continue

                x, y, w_box, h_box = cv2.boundingRect(cnt)
                if h_box <= 0:
                    continue

                aspect_ratio = float(w_box) / h_box
                if not (0.75 < aspect_ratio < 1.30):
                    continue

                # ROI extrahieren und aufbereiten
                roi_gray = gray[y:y + h_box, x:x + w_box]
                if roi_gray.size == 0:
                    continue

                roi_blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)
                dice_brightness = float(np.median(roi_blurred))
                threshold_value = max(35, min(int(dice_brightness * 0.72), 140))

                # Dunkle Augen (Pips) erkennen
                _, roi_thresh = cv2.threshold(roi_blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)
                pip_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                roi_thresh = cv2.morphologyEx(roi_thresh, cv2.MORPH_OPEN, pip_kernel)
                roi_thresh = cv2.morphologyEx(roi_thresh, cv2.MORPH_CLOSE, pip_kernel)
                eyes_debug[y:y + h_box, x:x + w_box] = roi_thresh

                # Auge-Konturen und Filterung
                pip_contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                raw_pips = []

                for p_cnt in pip_contours:
                    p_area = cv2.contourArea(p_cnt)
                    min_pip_size, max_pip_size = max(5, int(area * 0.003)), int(area * 0.08)
                    if not (min_pip_size <= p_area <= max_pip_size):
                        continue

                    # Zirkularitaet pruefen
                    p_perimeter = cv2.arcLength(p_cnt, True)
                    if p_perimeter <= 0:
                        continue
                    circularity = 4.0 * np.pi * p_area / (p_perimeter * p_perimeter)
                    if circularity < 0.35:
                        continue

                    # Aspectratio der Augenkontour
                    px, py, pw, ph = cv2.boundingRect(p_cnt)
                    if ph <= 0:
                        continue
                    pip_aspect = pw / float(ph)
                    if not (0.45 < pip_aspect < 2.2):
                        continue

                    # Schwerpunkt berechnen
                    M = cv2.moments(p_cnt)
                    if M["m00"] == 0:
                        continue
                    l_cx, l_cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    raw_pips.append((l_cx, l_cy))

                # Doppelte Augen entfernen (Distanz-Check)
                valid_pips = []
                min_dist_between_pips = max(6, int(min(w_box, h_box) * 0.12))
                for p in raw_pips:
                    if all(np.sqrt((p[0] - v[0]) ** 2 + (p[1] - v[1]) ** 2) > min_dist_between_pips for v in valid_pips):
                        valid_pips.append(p)

                # Position der Augen pruefen (muss im Wuerfel-Zentrum sein)
                final_pips = []
                roi_center_x, roi_center_y = w_box / 2.0, h_box / 2.0
                margin_x, margin_y = int(w_box * 0.06), int(h_box * 0.06)

                for l_cx, l_cy in valid_pips:
                    if not (margin_x < l_cx < w_box - margin_x and margin_y < l_cy < h_box - margin_y):
                        continue
                    dist_to_center = np.sqrt((l_cx - roi_center_x) ** 2 + (l_cy - roi_center_y) ** 2)
                    if dist_to_center > (min(w_box, h_box) * 0.52):
                        continue
                    final_pips.append((l_cx, l_cy))

                # Ergebnis visualisieren
                pip_count = len(final_pips)
                if 1 <= pip_count <= 6:
                    overall_score += pip_count
                    dice_results.append(pip_count)
                    for l_cx, l_cy in final_pips:
                        cv2.circle(display_frame, (l_cx + x, l_cy + y), 4, (255, 0, 0), -1)
                    cv2.rectangle(display_frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"W: {pip_count}", (x, max(20, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.rectangle(display_frame, (x, y), (x + w_box, y + h_box), (0, 255, 255), 1)
                    cv2.putText(display_frame, f"? ({pip_count})", (x, max(20, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

             # Debug-Ausgabe
            cv2.imshow("HSV Hintergrund-Maske", background_mask)
            cv2.imshow("Saettigungs-Wuerfel-Maske", dice_mask)

            cv2.imshow("Augen-Maske", eyes_debug)

            # Statistiken auf dem Bild anzeigen
            cv2.putText(display_frame, f"Wuerfel: {sorted(dice_results, reverse=True)}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Gesamtsumme: {overall_score}", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.imshow("RPi rpicam Multi-Dice Detector", display_frame)
            now = time.time()
            if now - last_print >= 0.5:
                print(f"Wuerfel: {sorted(dice_results, reverse=True)} | Gesamtsumme: {overall_score}")
                last_print = now

            # Exit bei 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nProgramm durch Ctrl+C beendet.")

    finally:
        print("Kamera wird beendet...")
        picam.stop()
        cv2.destroyAllWindows()
        print("Programm beendet.")



if __name__ == "__main__":
    detect_individual_dice()
