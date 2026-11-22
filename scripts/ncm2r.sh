#!/usr/bin/env bash
# ncm2r.sh (runs ON the Dell): NCM PHASE 2-RECON, "look in the pantry before cooking".
# Before the pantry audit (which codes/jars does a name like "Lily" use, and can we pull them?), print exactly how the
# released tokenizer stores its levels and codebooks, so the audit script is written against the real code, not guesses.
# Also downloads the TinyStories validation text (~20 MB) as the audit's test dishes. Read-only otherwise; ~1-2 min, little GPU.
set -uo pipefail
OUT=/data/lab/ncm2r; mkdir -p $OUT; exec > >(tee $OUT/REPORT.txt) 2>&1
cd /data/ncm/ncm; PY=/data/lab/venv_ncm/bin/python
echo "=== NCM PHASE 2-RECON $(date) ==="; git log -1 --format="commit %h %cd"
echo; echo "-- 1. repo layout"; find . -path ./artifacts -prune -o -path ./data -prune -o -path ./.git -prune -o -name "*.py" -print -o -name "*.yaml" -print | sort
echo; echo "-- 2. classes + functions in tokenization/ and models/"
grep -n -E "^\s*(class |def )" tokenization/*.py models/*.py 2>/dev/null
echo; echo "-- 3. how eval loads + calls the tokenizer"
grep -n -i -E "tokeni|vqvae|encode|decode|quantiz|codebook" scripts/*.py evaluate.py 2>/dev/null | head -80
echo; echo "-- 4. quantizer source (files mentioning codebook/quantiz)"
for f in $(grep -l -i -E "codebook|quantiz" tokenization/*.py models/*.py 2>/dev/null); do echo "##### $f ($(wc -l <$f) lines)"; head -400 "$f"; done
echo; echo "-- 5. configs"; for f in config/eval/tinystories.yaml config/eval/lm1b-small.yaml artifacts/tinystories/config.yaml artifacts/lm1b/small/config.yaml; do echo "##### $f"; cat $f 2>/dev/null; done
echo; echo "-- 6. what is inside tokenizer.pt"
$PY - <<'P'
import torch
for p in ["artifacts/tinystories/tokenizer.pt","artifacts/lm1b/tokenizer.pt"]:
    print("#####", p)
    try: o = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e: print("  load failed:", type(e).__name__, str(e)[:300]); continue
    print("  type:", type(o).__name__)
    sd = o if isinstance(o, dict) else getattr(o, "state_dict", lambda: {})()
    if isinstance(o, dict): print("  top keys:", list(o.keys())[:20])
    for k in ("state_dict","model","tokenizer","vqvae","ema"):
        if isinstance(o, dict) and isinstance(o.get(k), dict): sd = o[k]; print("  using sub-dict:", k)
    for k, v in sd.items():
        if hasattr(v, "shape") and any(s in k.lower() for s in ("code","embed","quant","level","scale")): print("   ", k, tuple(v.shape))
    if isinstance(o, dict):
        for k, v in o.items():
            if not hasattr(v, "shape") and not isinstance(v, dict): print("  meta", k, "=", str(v)[:200])
P
echo; echo "-- 7. test dishes: TinyStories validation text"
mkdir -p /data/ncm/audit
$PY - <<'P'
from huggingface_hub import hf_hub_download
import re, collections
try:
    p = hf_hub_download("roneneldan/TinyStories", "TinyStoriesV2-GPT4-valid.txt", repo_type="dataset", local_dir="/data/ncm/audit")
    t = open(p, encoding="utf-8", errors="ignore").read(); stories = [s.strip() for s in t.split("<|endoftext|>") if s.strip()]
    print("  stories:", len(stories), "| file:", p)
    c = collections.Counter(n for s in stories for n in set(re.findall(r"\b(Lily|Tim|Ben|Sue|Max|Tom|Mia|Sam|Anna|Jack|Lucy|Mom|Bob)\b", s)))
    print("  stories containing each name:", dict(c.most_common()))
except Exception as e: print("  !! download failed:", type(e).__name__, str(e)[:300])
P
echo; echo "=== RECON DONE. Tell Claude: ncm2r ready ==="
