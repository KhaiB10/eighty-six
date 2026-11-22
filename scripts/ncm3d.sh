#!/usr/bin/env bash
# ncm3d.sh (runs ON the Dell): NCM PHASE 3D, "which shelf to lock". The 2-round jar lock removed Lily but also turned many
# girl stories into boy/cat stories (any-girl 41.7% -> 25.7%). Hypothesis: the coarse shelves carry "a girl is the hero",
# the fine shelves carry "her name is Lily". Lock the same r2 jars, but only on some shelves, and see which removes Lily
# while keeping the girl. Arms: coarse L0-3 | middle L4-7 | fine L8-11 | middle+fine L4-11. ~9-10 min, full P40.
set -uo pipefail
OUT=/data/lab/ncm3d; mkdir -p $OUT; exec > >(tee $OUT/REPORT.txt) 2>&1
cd /data/ncm/ncm; PY=/data/lab/venv_ncm/bin/python
echo "=== NCM PHASE 3D: WHICH SHELF TO LOCK $(date) ==="
[ -f /data/lab/ncm3/r2/lily_mask.pt ] || { echo "!! /data/lab/ncm3/r2/lily_mask.pt missing (made by ncm3.sh)"; exit 1; }
echo "-- 1. split the r2 lock by shelf"
$PY - <<'P'
import torch
M = torch.load("/data/lab/ncm3/r2/lily_mask.pt")
for name, lv in {"coarse": range(0, 4), "middle": range(4, 8), "fine": range(8, 12), "midfine": range(4, 12)}.items():
    sub = {l: (v if l in lv else v[:0]) for l, v in M.items()}
    torch.save(sub, f"/data/lab/ncm3d/mask_{name}.pt"); print(f"   {name:<8} shelves {list(lv)}: {sum(len(v) for v in sub.values())} jars")
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
for a in coarse middle fine midfine; do run $a $OUT/mask_$a.pt; done
cp $OUT/eval_shard.py.orig scripts/eval_shard.py && echo && echo "-- eval_shard.py restored to original"
echo; echo "-- 2. who shows up (one count per story, 'hyps' field). Reference arms from ncm3/ncm3b included."
$PY - <<'P'
import glob, pickle, re, collections, random, statistics
random.seed(0)
GIRLS = ["Lola","Anna","Amy","Rose","Sue","Lucy","Mia","Jill","Ellie","Sara","Sarah","Kitty","Princess"]
w = lambda x, t: re.search(rf"\b{x}\b", t) is not None
ARMS = [("none","/data/lab/ncm3"),("r2 all","/data/lab/ncm3/r2x"),("ban","/data/lab/ncm3b/ban"),
        ("coarse","/data/lab/ncm3d/coarse"),("middle","/data/lab/ncm3d/middle"),("fine","/data/lab/ncm3d/fine"),("midfine","/data/lab/ncm3d/midfine")]
paths = {"none": "/data/lab/ncm3/none", "r2 all": "/data/lab/ncm3/r2", "ban": "/data/lab/ncm3b/ban"}
H = {}
for arm, _ in ARMS:
    d = paths.get(arm, f"/data/lab/ncm3d/{arm}")
    H[arm] = [h for f in sorted(glob.glob(f"{d}/shard_*.pkl")) for h in pickle.load(open(f, "rb"))["hyps"]]
print("arm      | Lily  | any girl (incl. Lily) | Tim   | 'she'/'her' stories | top names")
for arm, T in H.items():
    if not T: print(f"{arm:<8} | no outputs"); continue
    n = len(T); g = collections.Counter(x for t in T for x in GIRLS + ["Lily"] if w(x, t))
    she = sum(bool(re.search(r"\b(she|her)\b", t, re.I)) for t in T) / n * 100
    print(f"{arm:<8} | {sum(w('Lily',t) for t in T)/n*100:5.1f}% | {sum(any(w(x,t) for x in GIRLS+['Lily']) for t in T)/n*100:5.1f}%                | "
          f"{sum(w('Tim',t) for t in T)/n*100:5.1f}% | {she:5.1f}%              | {dict(g.most_common(4))}")
lily_i = [i for i, t in enumerate(H["none"]) if w("Lily", t)]
print(f"\nSame sample slots, cooked by each shelf lock (first 130 chars):")
for i in random.sample(lily_i, min(3, len(lily_i))):
    for arm in ("none", "coarse", "fine", "midfine"):
        if H.get(arm): print(f"  [{arm:<7}] {H[arm][i][:130].replace(chr(10), ' ')}")
    print()
P
echo "genPPL per arm is printed under each '-- arm:' line above (paper-style quality; none 16.39 | r2-all 16.47 | ban 17.14)"
echo "=== DONE. Tell Claude: ncm3d ready ==="
