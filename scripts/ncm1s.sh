#!/usr/bin/env bash
# ncm1s.sh (runs ON the Dell): NCM PHASE 1, the Sudoku part. Last time the test data wasn't where the config looked
# (data/sudoku/puzzles.pickle). Find where the puzzles really are, then run easy / medium / hard on the P40 in FP32.
# Paper (12 steps): easy 99.5%  medium 96.5%  hard 86.1% boards fully solved.
OUT=/data/lab/ncm1s; mkdir -p $OUT; exec > >(tee $OUT/REPORT.txt) 2>&1
echo "=== NCM PHASE 1: SUDOKU $(date) ==="
cd /data/ncm/ncm; PY=/data/lab/venv_ncm/bin/python
echo "-- where the Sudoku files are:"; find data/sudoku artifacts/sudoku -type f | sed 's/^/   /'
for lvl in easy medium hard; do
  P=$(find data/sudoku -name puzzles.pickle -path "*$lvl*" | head -1); [ -z "$P" ] && P=$(find data/sudoku -name "*$lvl*puzzles*.pickle" | head -1)
  DD=$(dirname "${P:-data/sudoku/x}")
  echo; echo "-- Sudoku $lvl (data: ${P:-NOT FOUND})"
  [ -z "$P" ] && continue
  t0=$(date +%s)
  timeout 3600 $PY evaluate.py --config config/eval/sudoku-$lvl.yaml --checkpoint-dir artifacts/sudoku/$lvl --data-dir $DD --out $OUT/sudoku-$lvl 2>&1 \
    | grep -vE "[Ww]arn|deprecat" | tail -6
  echo "   ($lvl: $(( $(date +%s) - t0 ))s)"; [ -f $OUT/sudoku-$lvl/shard_0.log ] && grep -i -m3 "error" $OUT/sudoku-$lvl/shard_0.log
done
echo; echo "paper: easy 99.5 | medium 96.5 | hard 86.1   (columns above: solved %, solved/n, cell %, valid-grid %)"
echo "=== DONE ==="
