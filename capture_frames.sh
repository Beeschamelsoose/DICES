#!/bin/bash

# Serienbilder mit rpicam-jpeg aufnehmen
# Für Würfel-Detektion

FRAMES=30          # Anzahl Frames
INTERVAL=100       # Millisekunden zwischen Frames
OUTPUT_DIR="frames"

echo "======================================"
echo "Dice Detection - Frame Capture"
echo "======================================"
echo "Frames: $FRAMES"
echo "Interval: ${INTERVAL}ms"
echo "Output: $OUTPUT_DIR/"
echo "======================================"

# Verzeichnis erstellen
mkdir -p $OUTPUT_DIR
cd $OUTPUT_DIR

# Alte Frames löschen
rm -f frame_*.jpg 2>/dev/null

# Capture Loop
echo ""
echo "Capturing frames..."
for i in $(seq -f "%04g" 1 $FRAMES); do
    echo -n "."
    rpicam-jpeg --timeout 100 --immediate --output frame_$i.jpg
    sleep $(echo "scale=3; $INTERVAL / 1000" | bc)
done

echo ""
echo ""
echo "======================================"
echo "✅ Capture complete!"
echo "======================================"
echo "Saved frames in: $OUTPUT_DIR/"
echo "Preview first frame:"
file frame_0001.jpg
ls -lh frame_000*.jpg | head -5
echo "..."
