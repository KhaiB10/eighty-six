#!/usr/bin/env bash
# ncm3b.sh (runs ON the Dell): NCM PHASE 3B. (1) Add the fair baseline: a plain WORD BAN ("never output the Lily token",
# applied at the very last step, like a normal LLM bad-words list). (2) Recount every arm properly: one count per real
# story, after printing what is actually inside the output files. Reuses ncm3's none / r2 / r8 outputs. ~4-5 min.
set -uo pipefail
OUT=/data/lab/ncm3b; mkdir -p $OUT; exec > >(tee $OUT/REPORT.txt) 2>&1
cd /data/ncm/ncm; PY=/data/lab/venv_ncm/bin/python
echo "=== NCM PHASE 3B: WORD-BAN BASELINE + PROPER RECOUNT $(date) ==="
for a in none r2 r8; do ls /data/lab/ncm3/$a/shard_*.pkl >/dev/null 2>&1 || { echo "!! /data/lab/ncm3/$a missing, run ncm3.sh first"; exit 1; }; done
echo "-- 1. patch the final word-picking step (guarded; inactive without NCM_BAN)"
cp scripts/eval_shard.py $OUT/eval_shard.py.orig
if ! grep -q "NCM_BAN_PATCH" scripts/eval_shard.py; then
  ln=$(grep -n "^net = NextConceptTransformer(" scripts/eval_shard.py | head -1 | cut -d: -f1)
  [ -z "$ln" ] && { echo "!! could not find where the cook is built"; exit 1; }
  head -n $((ln-1)) scripts/eval_shard.py > /tmp/es.py
  cat >> /tmp/es.py <<'PATCH'
# NCM_BAN_PATCH (Khai's ncm3b): plain word ban at the decoder's output. Inactive without NCM_BAN.
import os as _os2
if _os2.environ.get("NCM_BAN"):
    from models.vae import Decoder as _Dec
    _BAN = [int(x) for x in _os2.environ["NCM_BAN"].split(",")]
    _of = _Dec.forward
    def _f(self, z):
        out = _of(self, z).clone(); out[..., _BAN] = float("-inf"); return out
    _Dec.forward = _f
PATCH
  tail -n +$ln scripts/eval_shard.py >> /tmp/es.py; cp /tmp/es.py scripts/eval_shard.py
fi
$PY -m py_compile scripts/eval_shard.py && echo "   patched ok"
echo; echo "-- arm: ban (word 'Lily' = token ids 413, 674 forbidden at the last step)"; t0=$(date +%s)
NCM_BAN=413,674 timeout 3600 $PY evaluate.py --config config/eval/tinystories.yaml --checkpoint-dir artifacts/tinystories --out $OUT/ban 2>&1 \
  | grep -vE "[Ww]arn|deprecat|Loading weights|it/s\]" | tail -9
echo "   (ban: $(( $(date +%s) - t0 ))s)"
cp $OUT/eval_shard.py.orig scripts/eval_shard.py && echo "-- eval_shard.py restored to original"
echo; echo "-- 2. what is inside the output files, then one count per real story"
$PY - <<'P'
import sys, os, re, glob, pickle, collections
sys.path.insert(0, "/data/ncm/ncm"); os.chdir("/data/ncm/ncm")
from omegaconf import OmegaConf
from tokenization.vqvae_tokenizer import VQVAETokenizer
CK = "artifacts/tinystories"; CFG = OmegaConf.load(f"{CK}/config.yaml")
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(CK, CFG.checkpoints.tokenizer)
CFG.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(CK, CFG.checkpoints.bpe_vocab)
CFG.tokenizer.pre_tokenizer.merges_file_path = os.path.join(CK, CFG.checkpoints.bpe_merges)
pt = VQVAETokenizer.from_config(CFG.tokenizer).pre_tokenizer
def shape(o, d=0):
    if d > 3: return "..."
    if isinstance(o, dict): return "{" + ", ".join(f"{k!r}: {shape(v, d+1)}" for k, v in list(o.items())[:12]) + "}"
    if isinstance(o, (list, tuple)): return f"{type(o).__name__}[{len(o)}] of {shape(o[0], d+1) if len(o) else '-'}"
    if hasattr(o, "shape"): return f"{type(o).__name__}{tuple(o.shape)}"
    if isinstance(o, str): return f"str({len(o)})"
    return type(o).__name__
def as_text(v):
    if isinstance(v, str): return v
    if hasattr(v, "tolist"): v = v.tolist()
    if isinstance(v, (list, tuple)) and v and all(isinstance(t, int) for t in v):
        return pt.decode([t for t in v if 0 <= t < 8192], skip_special_tokens=True)
    return None
def load(arm, base):
    recs = []
    for f in sorted(glob.glob(f"{base}/{arm}/shard_*.pkl")): recs.append(pickle.load(open(f, "rb")))
    return recs
recs = load("none", "/data/lab/ncm3"); print("   file layout (arm none, shard 0):", shape(recs[0])[:600])
def samples(o):
    # find the list that has one entry per story (1280 total)
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, (list, tuple)) and len(v) in (256, 1280, 640): return v, k
        for v in o.values():
            r = samples(v)
            if r[0] is not None: return r
    if isinstance(o, (list, tuple)) and len(o) in (256, 1280, 640): return o, "(top-level list)"
    return None, None
GIRLS = ["Lola","Anna","Amy","Rose","Sue","Lucy","Mia","Jill","Ellie","Sara","Kitty"]
w = lambda x, t: re.search(rf"\b{x}\b", t) is not None
print("\n   arm  | field | stories | Lily | Tim | other girls | top stand-in names")
for arm, base in [("none", "/data/lab/ncm3"), ("r2", "/data/lab/ncm3"), ("r8", "/data/lab/ncm3"), ("ban", "/data/lab/ncm3b")]:
    per_field = collections.defaultdict(list)
    for r in load(arm, base):
        S, where = samples(r)
        if S is None: print(f"   {arm}: could not find the per-story list"); continue
        for s in S:
            if isinstance(s, dict):
                for k, v in s.items():
                    t = as_text(v)
                    if t: per_field[str(k)].append(t)
            else:
                t = as_text(s)
                if t: per_field[where].append(t)
    for k, T in per_field.items():
        n = len(T); g = collections.Counter(x for t in T for x in GIRLS if w(x, t))
        print(f"   {arm:<4} | {k[:14]:<14} | {n:>5} | {sum(w('Lily',t) for t in T)/n*100:5.1f}% | {sum(w('Tim',t) for t in T)/n*100:5.1f}% | "
              f"{sum(any(w(x,t) for x in GIRLS) for t in T)/n*100:5.1f}% | {dict(g.most_common(4))}")
    if arm == "ban" and per_field:
        T = max(per_field.values(), key=len)
        for t in [t for t in T if any(w(x, t) for x in GIRLS)][:1]: print("     ban sample:", t[:200].replace("\n", " "))
print("\n   genPPL so far (lower = better): none 16.39±0.33 | r2 16.47±0.34 | r8 17.81±0.43 | ban: see the table above")
P
echo; echo "=== DONE. Tell Claude: ncm3b ready ==="
