#!/usr/bin/env bash
# ------------------------------------------------------------------------
# run_spiral.sh  —  wrapper script to generate an outward‑growing spiral
#                  mosaic with cotono_kusokora.py (radial mosaic tool).
#
# Usage:
#   ./run_spiral.sh <IMAGE_PATH> [OUTPUT_PNG]
#
# Example:
#   ./run_spiral.sh \
#       "/Users/suzukiakiramuki/playground/被写体.png" \
#       spiral.png
# ------------------------------------------------------------------------
set -euo pipefail

# ---- parameters you can tweak -------------------------------------------
WEDGES=72            # number of radial wedges (smoothness)
SPACING=1.03         # >1.0 makes tiles spiral outward
TILE_ROT_STEP=5      # tile rotation increment = 360/WEDGES for perfect spiral
START=0.8            # offset (in tile widths) from center
TILE_SIZE=128        # resize every tile to this size (px)
CANVAS=3000          # canvas width & height (px)

SCRIPT="$(dirname "$0")/cotono_kusokora.py"   # assumes same folder
PYTHON="python"                                  # or python3 / py etc.

# ---- arg parsing ---------------------------------------------------------
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <IMAGE_PATH> [OUTPUT_PNG]" >&2
  exit 1
fi
IMG="$1"; shift
OUT="${1:-spiral.png}"

# ---- run ----------------------------------------------------------------
$PYTHON "$SCRIPT" "$IMG" \
  --wedges "$WEDGES" \
  --spacing "$SPACING" \
  --tile-rot-step "$TILE_ROT_STEP" \
  --start "$START" \
  --tile "$TILE_SIZE" \
  --size "$CANVAS" \
  --transparent \
  --out "$OUT"

echo "Spiral saved → $OUT"
