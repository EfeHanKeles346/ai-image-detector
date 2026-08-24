#!/bin/bash
# Outermost safety net: if fetch.py dies for any reason at all — crash, kill,
# unhandled exception — bring it straight back. Only the .ALL_DONE marker stops
# the loop. The previous run had nothing at this layer, so a single failure
# ended the whole night.
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ML_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
ROOT=${PIXELPROOF_DATA_ROOT:-"$ML_ROOT/data"}
PY=${PIXELPROOF_PYTHON:-"$ML_ROOT/.venv/bin/python"}
LOG=$ROOT/indirme.log

echo "[$(date '+%m-%d %H:%M:%S')] BEKÇİ BAŞLADI (pid $$)" >> "$LOG"
n=0
while [ ! -f "$ROOT/.ALL_DONE" ]; do
  n=$((n+1))
  echo "[$(date '+%m-%d %H:%M:%S')] --- fetch.py başlatılıyor (deneme $n) ---" >> "$LOG"
  "$PY" "$SCRIPT_DIR/fetch_datasets.py" >> "$LOG" 2>&1
  code=$?
  echo "[$(date '+%m-%d %H:%M:%S')] fetch.py çıktı (kod $code)" >> "$LOG"
  [ -f "$ROOT/.ALL_DONE" ] && break
  sleep 30
done
echo "[$(date '+%m-%d %H:%M:%S')] BEKÇİ BİTTİ — her şey indi" >> "$LOG"
