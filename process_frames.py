#!/usr/bin/env python3
"""
Würfel-Detektion aus Serienbildern
Basis-Verarbeitung mit OpenCV
"""

import cv2
import os
import numpy as np
from pathlib import Path

print("=" * 60)
print("Dice Detection - Frame Processing")
print("=" * 60)

# Verzeichnis mit Frames
frame_dir = "frames"

if not os.path.exists(frame_dir):
    print(f" Verzeichnis '{frame_dir}' nicht gefunden!")
    print("Erst: bash capture_frames.sh")
    exit(1)

# Alle Frame-Dateien finden
frames = sorted(Path(frame_dir).glob("frame_*.jpg"))

if not frames:
    print(f" Keine Frames in '{frame_dir}' gefunden!")
    exit(1)

print(f"\n[OK] {len(frames)} Frames gefunden\n")

# Verarbeitungsparameter
MIN_CONTOUR_AREA = 500  # Mindestgröße für Würfel
BLUR_SIZE = (5, 5)

# Verarbeitungsergebnisse speichern
output_dir = "processed"
os.makedirs(output_dir, exist_ok=True)

print("Processing frames...")
print("-" * 60)

for idx, frame_path in enumerate(frames):
    # Frame laden
    img = cv2.imread(str(frame_path))
    if img is None:
        print(f" Konnte {frame_path} nicht laden")
        continue
    
    # Bildinfo
    h, w = img.shape[:2]
    
    # 1. In Graustufen konvertieren
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Glätten
    blurred = cv2.GaussianBlur(gray, BLUR_SIZE, 0)
    
    # 3. Kantenerkennung (Canny)
    edges = cv2.Canny(blurred, 50, 150)
    
    # 4. Konturen finden
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5. Große Konturen filtern (potenzielle Würfel)
    filtered_contours = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]
    
    # Visualisierung
    result_img = img.copy()
    
    # Zeichne gefundene Würfel
    for contour in filtered_contours:
        x, y, w_rect, h_rect = cv2.boundingRect(contour)
        cv2.rectangle(result_img, (x, y), (x + w_rect, y + h_rect), (0, 255, 0), 2)
    
    # Info-Text
    dice_count = len(filtered_contours)
    cv2.putText(result_img, f"Dice: {dice_count}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(result_img, f"Frame {idx+1}/{len(frames)}", (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    
    # Speichern
    frame_name = frame_path.stem
    output_path = os.path.join(output_dir, f"{frame_name}_processed.jpg")
    cv2.imwrite(output_path, result_img)
    
    print(f"[{idx+1:2d}/{len(frames)}] {frame_name}: {dice_count} objects found")

print("-" * 60)
print(f"\n Verarbeitung abgeschlossen!")
print(f"Ergebnisse in: {output_dir}/\n")

# Zeige erstes verarbeitetes Bild
first_result = os.path.join(output_dir, f"frame_0001_processed.jpg")
if os.path.exists(first_result):
    print("Preview:")
    #cv2.imshow("First Result", first_result)
    print("(Drücke beliebige Taste zum Schließen)")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("=" * 60)
