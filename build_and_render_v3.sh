#!/usr/bin/env bash
set -e

SCENE_FILE="${1:-ours_cook_spinach.4dgs}"
OUT_DIR="${2:-frames}"
FRAME_COUNT="${3:-120}"
WIDTH="${4:-1280}"
HEIGHT="${5:-720}"

echo "=== Building CPU 4DGS preview renderer ==="

if [ ! -f "temporal_4dgs_decoder.c" ]; then
  echo "ERROR: temporal_4dgs_decoder.c not found in this folder."
  echo "Put temporal_4dgs_decoder.c here first."
  exit 1
fi

if [ ! -f "temporal_4dgs_decoder.h" ]; then
  echo "ERROR: temporal_4dgs_decoder.h not found in this folder."
  echo "Put temporal_4dgs_decoder.h here first."
  exit 1
fi

if [ ! -f "cpu_4dgs_preview_v3.c" ]; then
  echo "ERROR: cpu_4dgs_preview_v3.c not found in this folder."
  exit 1
fi

if [ ! -f "$SCENE_FILE" ]; then
  echo "ERROR: scene file not found: $SCENE_FILE"
  exit 1
fi

gcc -std=c99 -O2 temporal_4dgs_decoder.c cpu_4dgs_preview_v3.c -o cpu_4dgs_preview -lm

echo "=== Rendering frames ==="
./cpu_4dgs_preview "$SCENE_FILE" "$OUT_DIR" "$FRAME_COUNT" "$WIDTH" "$HEIGHT"

if command -v ffmpeg >/dev/null 2>&1; then
  echo "=== Creating MP4 video ==="
  ffmpeg -y -framerate 30 -i "$OUT_DIR/frame_%04d.ppm" -c:v libx264 -pix_fmt yuv420p output_preview.mp4
  echo "Done: output_preview.mp4"
else
  echo "ffmpeg not found."
  echo "Install it with:"
  echo "  brew install ffmpeg"
  echo "Then run:"
  echo "  ffmpeg -y -framerate 30 -i $OUT_DIR/frame_%04d.ppm -c:v libx264 -pix_fmt yuv420p output_preview.mp4"
fi