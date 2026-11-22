#!/usr/bin/env bash
# ncm4.sh (runs ON the Dell): eighty-six PHASE 4, "tune the middle lock". Phase 3d showed the name lives on the middle
# shelves (L4-7). Here the audit itself is restricted to L4-7: each round, re-check where Lily went and lock the next 32
# Lily jars per middle shelf. Rounds 1-4 each become a cook arm, plus each round + the fine-shelf jars from r2 (L8-11).
# Goal: Lily -> 0% while girl-hero stories stay near 42%. ~20 min, full P40.
set -uo pipefail
OUT=/data/lab/ncm4; mkdir -p $OUT/pantry; exec > >(tee $OUT/REPORT.txt) 2>&1
cd /data/ncm/ncm; PY=/data/lab/venv_ncm/bin/python
echo "=== eighty-six PHASE 4: TUNE THE MIDDLE LOCK $(date) ==="
for f in /data/lab/ncm2b.sh /data/lab/ncm3/r2/lily_mask.pt; do [ -f $f ] || { echo "!! missing $f"; exit 1; }; done
echo "-- 1. pantry rounds, middle shelves only (L4-7), 32 new jars per shelf per round"
$PY - <<'Q'
s = open("/data/lab/ncm2b.sh").read()
s = s.replace("K, ROUNDS, ROWS = 32, 8, []", "K, ROUNDS, ROWS = 32, 4, []")
s = s.replace("/data/lab/ncm2b", "/data/lab/ncm4/pantry")
old = "    for l in range(len(LEVELS)):\n        codes = level_codes(si, l)[b_idx, pos_to_slot(l, p_idx)]"
assert old in s, "ncm2b layout changed"
s = s.replace(old, "    for l in range(4, 8):\n        codes = level_codes(si, l)[b_idx, pos_to_slot(l, p_idx)]")
old2 = "    if gone >= 0.98:"
assert old2 in s
s = s.replace(old2, "    torch.save({l: torch.nonzero(masked[l]).flatten().cpu() for l in masked}, f\"{OUT}/mask_mid_r{r}.pt\")\n" + old2)
cut = s.index('echo; echo "-- COOK RECON')
s = s[:cut] + 'echo done\n'
open("/data/lab/ncm4/pantry/make.sh", "w").write(s)
Q
bash $OUT/pantry/make.sh > $OUT/pantry/log.txt 2>&1
grep -E "^\s+[0-9] \|" $OUT/pantry/log.txt || { tail -20 $OUT/pantry/log.txt; exit 1; }
$PY - <<'P'
import torch, glob
fine = torch.load("/data/lab/ncm3/r2/lily_mask.pt")
for f in sorted(glob.glob("/data/lab/ncm4/pantry/mask_mid_r*.pt")):
    m = torch.load(f); r = f.split("_r")[-1][:-3]
    mf = {l: (torch.unique(torch.cat([m[l], fine[l]])) if l >= 8 else m[l]) for l in m}
    torch.save(mf, f"/data/lab/ncm4/mask_midfine_r{r}.pt"); torch.save(m, f"/data/lab/ncm4/mask_mid_r{r}.pt")
    print(f"   round {r}: middle-only {sum(len(v) for v in m.values())} jars | + fine {sum(len(v) for v in mf.values())} jars")
P
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
    | grep -vE "[Ww]arn|deprecat|Loading weights|it/s\]" | tail -3
  echo "   ($1: $(( $(date +%s) - t0 ))s)"
}
ARMS=""; for r in 1 2 3 4; do for k in mid midfine; do [ -f $OUT/mask_${k}_r$r.pt ] && { run ${k}_r$r $OUT/mask_${k}_r$r.pt; ARMS="$ARMS ${k}_r$r"; }; done; done
export ARMS
cp $OUT/eval_shard.py.orig scripts/eval_shard.py && echo && echo "-- eval_shard.py restored to original"
echo; echo "-- 2. who shows up (one count per story, 'hyps' field). Reference arms from ncm3/ncm3b included."
$PY - <<'P'
import glob, pickle, re, collections, random, statistics
random.seed(0)
GIRLS = ["Lola","Anna","Amy","Rose","Sue","Lucy","Mia","Jill","Ellie","Sara","Sarah","Kitty","Princess"]
w = lambda x, t: re.search(rf"\b{x}\b", t) is not None
import os
ARMS = [(a, None) for a in ["none", "ban", "3d middle", "3d midfine"] + os.environ.get("ARMS", "").split()]
paths = {"none": "/data/lab/ncm3/none", "ban": "/data/lab/ncm3b/ban", "3d middle": "/data/lab/ncm3d/middle", "3d midfine": "/data/lab/ncm3d/midfine"}
H = {}
for arm, _ in ARMS:
    d = paths.get(arm, f"/data/lab/ncm4/{arm}")
    H[arm] = [h for f in sorted(glob.glob(f"{d}/shard_*.pkl")) for h in pickle.load(open(f, "rb"))["hyps"]]
print("arm         | Lily  | any girl (incl. Lily) | Tim   | 'she'/'her' stories | top names")
for arm, T in H.items():
    if not T: print(f"{arm:<11} | no outputs"); continue
    n = len(T); g = collections.Counter(x for t in T for x in GIRLS + ["Lily"] if w(x, t))
    she = sum(bool(re.search(r"\b(she|her)\b", t, re.I)) for t in T) / n * 100
    print(f"{arm:<11} | {sum(w('Lily',t) for t in T)/n*100:5.1f}% | {sum(any(w(x,t) for x in GIRLS+['Lily']) for t in T)/n*100:5.1f}%                | "
          f"{sum(w('Tim',t) for t in T)/n*100:5.1f}% | {she:5.1f}%              | {dict(g.most_common(4))}")
lily_i = [i for i, t in enumerate(H["none"]) if w("Lily", t)]
print(f"\nSame sample slots, cooked by each shelf lock (first 130 chars):")
for i in random.sample(lily_i, min(3, len(lily_i))):
    for arm in ["none", "3d midfine"] + os.environ.get("ARMS", "").split()[-2:]:
        if H.get(arm): print(f"  [{arm:<11}] {H[arm][i][:130].replace(chr(10), ' ')}")
    print()
P
echo "genPPL per arm is printed under each '-- arm:' line above (none 16.39 | ban 17.14 | 3d middle 16.56 | 3d midfine 16.73)"
echo "=== DONE. Tell Claude: ncm4 ready ==="
