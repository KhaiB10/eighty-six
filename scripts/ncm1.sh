#!/usr/bin/env bash
# ncm1.sh (runs ON the Dell) [v2: downloads now visible, with Python fallback]: NCM REPLICATION, PHASE 1. Run Stanford's released Next-Concept Modeling checkpoints on
# a 2016 Tesla P40 in plain FP32 (the paper used BF16 + torch.compile on modern GPUs) and compare with the paper.
#   Sudoku easy/medium/hard (paper: 99.5 / 96.5 / 86.1 % solved, 12 steps)
#   TinyStories           (paper: gen. perplexity 15.41, entropy 3.98, 12 steps)
#   LM1B small (97M)      (paper: gen. perplexity 93.05, 12 steps)
# Code: github.com/HazyResearch/ncm (MIT). Weights: huggingface hazyresearch/ncm-checkpoints. ~2 GB download + GPT-2 large
# (~3 GB) for scoring. Nothing else on the Dell is touched; it pauses nothing (Aster keeps living, using little GPU).
set -uo pipefail
N=/data/ncm; OUT=/data/lab/ncm1; mkdir -p "$N" "$OUT"; exec > >(tee "$OUT/REPORT.txt") 2>&1
echo "=== NCM PHASE 1: THEIR MODELS ON OUR P40 $(date) ==="
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
echo "-- 1. code"
[ -d $N/ncm/.git ] || git clone -q https://github.com/HazyResearch/ncm.git $N/ncm
cd $N/ncm && git log -1 --format="   commit %h  %cd" && cd - >/dev/null
echo "-- 2. python environment (copy of the working P40 venv + NCM's needs)"
if [ ! -x /data/lab/venv_ncm/bin/python ]; then cp -a /data/lab/venv2 /data/lab/venv_ncm && echo "   venv copied"; fi
PY=/data/lab/venv_ncm/bin/python
$PY -m pip install -q omegaconf hydra-core "transformers>=4.51" "huggingface_hub[cli]" regex rouge_score sacrebleu 2>&1 | grep -v -i "warn" | tail -3
$PY -c "import torch;print('   torch',torch.__version__,'| GPU',torch.cuda.get_device_name(0),'| bf16 supported:',torch.cuda.is_bf16_supported())"
HF=/data/lab/venv_ncm/bin/hf
echo "-- 3. checkpoints + test sets"
cd $N/ncm
$PY - <<'P'
from huggingface_hub import snapshot_download
for repo, kind, pats, dest in [("hazyresearch/ncm-checkpoints", "model", ["sudoku/*", "tinystories/*", "lm1b/tokenizer.pt", "lm1b/small/*"], "artifacts"),
                               ("hazyresearch/ncm-datasets", "dataset", ["sudoku/*"], "data")]:
    try:
        p = snapshot_download(repo_id=repo, repo_type=kind, allow_patterns=pats, local_dir=dest)
        print("   downloaded", repo, "->", p)
    except Exception as e:
        print("   !! download failed:", repo, type(e).__name__, str(e)[:300])
P
echo "   files:"; find artifacts data -type f 2>/dev/null | sed 's/^/     /' | head -40; du -sh artifacts data 2>/dev/null | sed 's/^/   /'
run() {  # name config ckptdir [datadir]
  local t0=$(date +%s); echo; echo "-- $1"
  timeout 7200 $PY evaluate.py --config config/eval/$2.yaml --checkpoint-dir artifacts/$3 ${4:+--data-dir data/$4} --out $OUT/$2 2>&1 \
    | grep -vE "[Ww]arn|deprecat" | tail -12
  local rc=${PIPESTATUS[0]}; echo "   ($1: exit $rc, $(( $(date +%s) - t0 ))s)"
  [ $rc -ne 0 ] && tail -15 $OUT/$2/shard_0.log
}
echo; echo "-- 4. EVALUATION (P40, FP32)  peak GPU memory is sampled below"
( while sleep 5; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader >> $OUT/gpu.log; done ) & MON=$!
run "Sudoku EASY   (paper 99.5%)" sudoku-easy   sudoku/easy   sudoku
run "Sudoku MEDIUM (paper 96.5%)" sudoku-medium sudoku/medium sudoku
run "Sudoku HARD   (paper 86.1%)" sudoku-hard   sudoku/hard   sudoku
run "TinyStories   (paper genPPL 15.41, entropy 3.98)" tinystories tinystories
run "LM1B small    (paper genPPL 93.05)" lm1b-small lm1b/small
kill $MON 2>/dev/null
echo; echo "-- GPU: peak memory $(sort -t, -k1 -n $OUT/gpu.log | tail -1 | cut -d, -f1), peak load $(sort -t, -k2 -n $OUT/gpu.log | tail -1 | cut -d, -f2)"
echo "=== NCM PHASE 1 DONE. Tell Claude: ncm1 ready ==="
