#!/usr/bin/env bash
# ncm13.sh (runs ON the Dell): eighty-six SCALE-UP STEP 1, "meet the three cooks". Same pantry, three cook sizes
# (LM1B small 97M / medium 317M / large 725M, real news sentences). No locks yet:
#   (1) replicate Stanford's quality numbers for medium + large on the P40 (small was done in Phase 1: 93.15 vs 93.05),
#       and time each one, so we know how long the lock tests will take;
#   (2) download the real LM1B validation sentences for the audits;
#   (3) count how often candidate targets (names, places, topics) appear in real text AND in each cook's sentences,
#       so we only lock targets common enough to measure.
# Start with ncm13go.sh (detached). Probably 30-60 min (the large cook is ~8x the TinyStories cook).
set -uo pipefail
OUT=/data/lab/ncm13; mkdir -p $OUT; exec > >(tee $OUT/REPORT.txt) 2>&1
cd /data/ncm/ncm; PY=/data/lab/venv_ncm/bin/python
echo "=== eighty-six SCALE-UP STEP 1: MEET THE THREE COOKS $(date) ==="
OLLAMA_STOPPED=0
if systemctl is-active --quiet ollama 2>/dev/null; then
  if sudo -n systemctl stop ollama 2>/dev/null; then OLLAMA_STOPPED=1; echo "   paused Ollama (Aster) for this run; it restarts automatically at the end"
  else echo "   !! could not pause Ollama without a password; run 'sudo systemctl stop ollama' first or Aster may crowd the GPU"; fi
fi
trap '[ $OLLAMA_STOPPED = 1 ] && sudo -n systemctl start ollama && echo "   Ollama (Aster) restarted"' EXIT
for m in $(curl -s -m 5 http://localhost:11434/api/ps 2>/dev/null | grep -o '"name":"[^"]*"' | cut -d'"' -f4); do
  curl -s -m 20 http://localhost:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null && echo "   unloaded $m"
done
sleep 3; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/   GPU before start: /'



echo "-- 1. real LM1B validation sentences"
$PY - <<'P'
from huggingface_hub import hf_hub_download
for f in ["lm1b/valid_bert_30522.bin", "lm1b/valid_bert_30522_offsets.npy", "lm1b/README.md"]:
    try: print("   ok:", hf_hub_download("hazyresearch/ncm-datasets", f, repo_type="dataset", local_dir="/data/ncm/audit"))
    except Exception as e: print("   !!", f, type(e).__name__, str(e)[:200])
P
head -c 1500 /data/ncm/audit/lm1b/README.md 2>/dev/null | sed 's/^/   | /'
run() {  # name config ckptdir
  local t0=$(date +%s); echo; echo "-- 2. quality: $1"
  timeout 14400 $PY evaluate.py --config config/eval/$2.yaml --checkpoint-dir artifacts/$3 --out $OUT/$2 2>&1 \
    | grep -vE "[Ww]arn|deprecat|Loading weights|it/s\]" | tail -8
  echo "   ($1: $(( ($(date +%s) - t0) / 60 )) min)"
}
run "LM1B MEDIUM (317M)" lm1b-medium lm1b/medium
run "LM1B LARGE  (725M)" lm1b-large  lm1b/large
echo; echo "   (small, from Phase 1 on this P40: genPPL 93.15 +/- 2.37, paper 93.05)"
echo; echo "-- 3. how common are candidate targets? (% of sentences containing the word)"
$PY - <<'P'
import numpy as np, glob, pickle, re, os
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
CANDS = ["obama", "bush", "china", "iraq", "afghanistan", "london", "paris", "america", "police", "president", "government",
         "election", "football", "game", "team", "season", "music", "film", "bank", "market", "oil", "economy", "war", "health", "school", "company"]
def rate(texts):
    texts = [t.lower() for t in texts]; return {w: sum(bool(re.search(rf"\b{w}\b", t)) for t in texts) / max(len(texts), 1) * 100 for w in CANDS}
real = []
try:
    ids = np.fromfile("/data/ncm/audit/lm1b/valid_bert_30522.bin", dtype=np.uint16)
    off = np.load("/data/ncm/audit/lm1b/valid_bert_30522_offsets.npy")
    print(f"   real validation: {len(off)} sentences, {len(ids)} tokens, offsets shape {off.shape}, first offsets {off[:4].tolist()}")
    starts = off[:, 0] if off.ndim == 2 else off
    ends = off[:, 1] if off.ndim == 2 else np.append(off[1:], len(ids))
    pick = np.random.default_rng(0).choice(len(starts), size=min(20000, len(starts)), replace=False)
    real = [tok.decode(ids[starts[i]:ends[i]], skip_special_tokens=True) for i in pick]
    print("   two real sentences:", real[0][:120], "|", real[1][:120])
except Exception as e: print("   !! could not read the validation file:", type(e).__name__, str(e)[:200])
def load(d): return [h for p in sorted(glob.glob(f"{d}/shard_*.pkl")) for h in pickle.load(open(p, "rb"))["hyps"]]
G = {"small": load("/data/lab/ncm1/lm1b-small"), "medium": load("/data/lab/ncm13/lm1b-medium"), "large": load("/data/lab/ncm13/lm1b-large")}
R = {"REAL text": rate(real)} if real else {}
for k, v in G.items():
    if v: R[f"{k} cook"] = rate(v); print(f"   {k} cook: {len(v)} sentences, e.g. {v[0][:110]!r}")
cols = list(R)
print("\n   word         | " + " | ".join(f"{c:>10}" for c in cols))
for w in sorted(CANDS, key=lambda w: -R[cols[-1]][w] if cols else 0):
    print(f"   {w:<12} | " + " | ".join(f"{R[c][w]:9.1f}%" for c in cols))
print("\n   A target needs ~3%+ in the cooks' sentences to be measurable with 1,280 samples (about 40 sentences).")
P
echo "=== DONE. Tell Claude: ncm13 ready ==="
