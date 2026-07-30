#!/bin/bash
# Outermost safety net: if fetch.py dies for any reason at all — crash, kill,
# unhandled exception — bring it straight back. Only the .ALL_DONE marker stops
# the loop. The previous run had nothing at this layer, so a single failure
# ended the whole night.
ROOT=/Volumes/LaCie/pixelproof-datasets
PY=/Users/efehankeles/Desktop/ai-image-detector/ml/.venv/bin/python
LOG=$ROOT/indirme.log

echo "[$(date '+%m-%d %H:%M:%S')] BEKÇİ BAŞLADI (pid $$)" >> "$LOG"
n=0
while [ ! -f "$ROOT/.ALL_DONE" ]; do
  n=$((n+1))
  echo "[$(date '+%m-%d %H:%M:%S')] --- fetch.py başlatılıyor (deneme $n) ---" >> "$LOG"
  "$PY" "$ROOT/fetch.py" >> "$LOG" 2>&1
  code=$?
  echo "[$(date '+%m-%d %H:%M:%S')] fetch.py çıktı (kod $code)" >> "$LOG"
  [ -f "$ROOT/.ALL_DONE" ] && break
  sleep 30
done
echo "[$(date '+%m-%d %H:%M:%S')] BEKÇİ BİTTİ — her şey indi" >> "$LOG"
