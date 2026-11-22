#!/usr/bin/env bash
# ncm3.sh (runs ON the Dell): NCM PHASE 3, "the cook test". Does the GENERATOR forget Lily if it is forbidden from
# picking Lily's jars while it writes? Three arms, same official TinyStories eval (1,280 stories, 5 seeds):
#   none = normal cook | r2 = 2 rounds of pulled jars (760, the sweet spot) | r8 = 8 rounds (3,064 jars, the full ncm2b mask)
# Measures: % of stories that mention Lily, Tim (control), other girl names; plus the paper's genPPL + entropy (quality).
# How: a small guarded patch in scripts/eval_shard.py sets forbidden jars to -inf before each level is sampled. The
# patch does nothing unless NCM_MASK is set, and the original file is restored at the end. ~12-15 min, full P40.
set -uo pipefail
OUT=/data/lab/ncm3; mkdir -p $OUT/r2; exec > >(tee $OUT/REPORT.txt) 2>&1
cd /data/ncm/ncm; PY=/data/lab/venv_ncm/bin/python
echo "=== NCM PHASE 3: COOK TEST $(date) ==="
echo "-- 1. rebuild the 2-round mask (same recipe as ncm2b, stopped at round 2)"
[ -f /data/lab/ncm2b/lily_mask.pt ] || { echo "!! /data/lab/ncm2b/lily_mask.pt missing, run ncm2b.sh first"; exit 1; }
sed -e 's/K, ROUNDS, ROWS = 32, 8, \[\]/K, ROUNDS, ROWS = 32, 2, []/' -e 's#/data/lab/ncm2b#/data/lab/ncm3/r2#g' \
    -e '/^echo; echo "-- COOK RECON/,/^grep -n/d' /data/lab/ncm2b.sh > $OUT/r2/make_r2.sh
bash $OUT/r2/make_r2.sh > $OUT/r2/log.txt 2>&1; grep -E "^\s+[12] \|" $OUT/r2/log.txt
[ -f $OUT/r2/lily_mask.pt ] || { echo "!! round-2 mask failed:"; tail -20 $OUT/r2/log.txt; exit 1; }
echo "-- 2. patch the cook (guarded; inactive without NCM_MASK)"
cp scripts/eval_shard.py $OUT/eval_shard.py.orig
if ! grep -q "NCM_MASK_PATCH" scripts/eval_shard.py; then
  ln=$(grep -n "^net = NextConceptTransformer(" scripts/eval_shard.py | head -1 | cut -d: -f1)
  [ -z "$ln" ] && { echo "!! could not find where the cook is built"; exit 1; }
  head -n $((ln-1)) scripts/eval_shard.py > /tmp/es.py
  cat >> /tmp/es.py <<'PATCH'
# NCM_MASK_PATCH (Khai's ncm3 cook test): forbid masked jars at each level while sampling. Inactive without NCM_MASK.
import os as _os
if _os.environ.get("NCM_MASK"):
    import torch as _t
    from models.next_concept import NextConceptTransformer as _NCT
    _MASK = {int(k): v.long() for k, v in _t.load(_os.environ["NCM_MASK"]).items()}
    _HITS = {"l0_masked": 0, "l0_total": 0}
    _og, _os_ = _NCT.generate, _NCT._sample_level
    def _gen(self, idx, *a, **k):
        if idx is not None and 0 in _MASK:
            _HITS["l0_total"] += idx.numel(); _HITS["l0_masked"] += _t.isin(idx, _MASK[0].to(idx.device)).sum().item()
        self._lvl = 1 if idx is not None else 0
        return _og(self, idx, *a, **k)
    def _samp(self, logits, *a, **k):
        l = getattr(self, "_lvl", 0); self._lvl = l + 1; c = _MASK.get(l)
        if c is not None and len(c):
            logits = logits.clone(); logits[..., c.to(logits.device)] = float("-inf")
        return _os_(self, logits, *a, **k)
    _NCT.generate, _NCT._sample_level = _gen, _samp
    import atexit; atexit.register(lambda: print("NCM_MASK level-0 picks that were masked jars:", _HITS))
PATCH
  tail -n +$ln scripts/eval_shard.py >> /tmp/es.py; cp /tmp/es.py scripts/eval_shard.py
fi
$PY -m py_compile scripts/eval_shard.py && echo "   patched ok"
run() {  # arm maskfile
  local t0=$(date +%s); echo; echo "-- arm: $1"
  NCM_MASK=$2 timeout 3600 $PY evaluate.py --config config/eval/tinystories.yaml --checkpoint-dir artifacts/tinystories --out $OUT/$1 2>&1 \
    | grep -vE "[Ww]arn|deprecat|Loading weights" | tail -9
  echo "   ($1: $(( $(date +%s) - t0 ))s)"; grep -h "NCM_MASK" $OUT/$1/shard_*.log 2>/dev/null
}
run none ""
run r2 $OUT/r2/lily_mask.pt
run r8 /data/lab/ncm2b/lily_mask.pt
cp $OUT/eval_shard.py.orig scripts/eval_shard.py && echo && echo "-- eval_shard.py restored to original"
echo; echo "-- 3. what the cook wrote: who shows up in the stories"
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
def texts(o):
    if isinstance(o, str): yield o
    elif isinstance(o, dict):
        for v in o.values(): yield from texts(v)
    elif isinstance(o, (list, tuple)):
        if o and all(isinstance(t, int) for t in o): yield pt.decode(list(o), skip_special_tokens=True)
        else:
            for v in o: yield from texts(v)
    elif hasattr(o, "tolist"): yield from texts(o.tolist())
GIRLS = ["Lola","Anna","Amy","Rose","Sue","Lucy","Mia","Jill","Ellie","Sara","Kitty"]
print("   arm  | stories | Lily | Tim (control) | other girl names | Lily's stand-ins")
for arm in ["none", "r2", "r8"]:
    T = []
    for f in sorted(glob.glob(f"/data/lab/ncm3/{arm}/shard_*.pkl")): T += [t for t in texts(pickle.load(open(f, "rb"))) if len(t) > 40]
    if not T: print(f"   {arm:<4} | no outputs found"); continue
    has = lambda w: sum(bool(re.search(rf"\b{w}\b", t)) for t in T) / len(T) * 100
    g = collections.Counter(w for t in T for w in GIRLS if re.search(rf"\b{w}\b", t))
    girl = sum(any(re.search(rf"\b{w}\b", t) for w in GIRLS) for t in T) / len(T) * 100
    print(f"   {arm:<4} | {len(T):>7} | {has('Lily'):5.1f}% | {has('Tim'):5.1f}% | {girl:5.1f}% | {dict(g.most_common(5))}")
    if arm == "r8":
        for t in [t for t in T if "Lily" in t][:2]: print("     sample still mentioning Lily:", t[:220].replace("\n", " "))
P
echo; echo "=== DONE. Tell Claude: ncm3 ready ==="
