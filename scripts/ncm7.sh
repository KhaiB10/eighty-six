#!/usr/bin/env bash
# ncm7.sh (runs ON the Dell): eighty-six STEP 1 of the ladder, "86 Spot". Repeat the Lily experiment with a different
# name (Spot, a dog) to check it wasn't luck, and as the bottom rung of the specificity ladder (animal > dog > Spot).
# PREDICTION (written before running): like Lily, middle-shelf locks remove Spot while dog stories survive with another
# dog name; top-shelf locks change the story itself. Same recipe as Phase 3d: 2 audit rounds, 32 jars/shelf/round,
# then lock by shelf group. Start it with ncm7go.sh (detached). ~15 min.
set -uo pipefail
OUT=/data/lab/ncm7; mkdir -p $OUT/pantry; exec > >(tee $OUT/REPORT.txt) 2>&1
cd /data/ncm/ncm; PY=/data/lab/venv_ncm/bin/python
echo "=== eighty-six LADDER STEP 1: 86 SPOT $(date) ==="
[ -f /data/lab/ncm2b.sh ] && [ -d /data/lab/ncm3/none ] || { echo "!! need /data/lab/ncm2b.sh and /data/lab/ncm3/none"; exit 1; }
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

echo "-- 1. how common is Spot?"
grep -o -w "Spot" /data/ncm/audit/TinyStoriesV2-GPT4-valid.txt | wc -l | sed 's/^/   "Spot" appears this many times in the validation set: /'
echo "-- 2. pantry audit, 2 rounds (Lily recipe, name = Spot)"
$PY - <<'Q'
s = open("/data/lab/ncm2b.sh").read()
for old, new in [('NAME, CONTROL = "Lily", "Tim"', 'NAME, CONTROL = "Spot", "Tim"'),
                 ("K, ROUNDS, ROWS = 32, 8, []", "K, ROUNDS, ROWS = 32, 2, []"),
                 ('name_score, name_test = got["name"][:1500], got["name"][1500:]',
                  'cut = int(len(got["name"]) * 0.75); name_score, name_test = got["name"][:cut], got["name"][cut:]'),
                 ("/data/lab/ncm2b", "/data/lab/ncm7/pantry")]:
    assert old in s, f"ncm2b layout changed: {old[:40]}"; s = s.replace(old, new)
s = s[:s.index('echo; echo "-- COOK RECON')] + "echo pantry-done\n"
open("/data/lab/ncm7/pantry/make.sh", "w").write(s)
Q
bash $OUT/pantry/make.sh > $OUT/pantry/log.txt 2>&1
grep -E "token ids that are|name split|BASELINE|kept in rebuilt|^\s+[12] \|" $OUT/pantry/log.txt | sed 's/^/   /'
[ -f $OUT/pantry/lily_mask.pt ] || { echo "!! pantry step failed:"; tail -15 $OUT/pantry/log.txt; exit 1; }
$PY - <<'P'
import torch
M = torch.load("/data/lab/ncm7/pantry/lily_mask.pt")   # (file name kept from the Lily recipe; it holds Spot's jars)
for name, lv in {"coarse": range(0, 4), "middle": range(4, 8), "fine": range(8, 12), "midfine": range(4, 12), "all": range(0, 12)}.items():
    sub = {l: (v if l in lv else v[:0]) for l, v in M.items()}
    torch.save(sub, f"/data/lab/ncm7/mask_{name}.pt"); print(f"   {name:<8} {sum(len(v) for v in sub.values())} jars")
P
echo "-- 3. patch the cook (guarded; inactive without NCM_MASK)"
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


run() {
  local t0=$(date +%s); echo; echo "-- arm: $1"
  NCM_MASK=$2 timeout 3600 $PY evaluate.py --config config/eval/tinystories.yaml --checkpoint-dir artifacts/tinystories --out $OUT/$1 2>&1 \
    | grep -vE "[Ww]arn|deprecat|Loading weights|it/s\]" | tail -1
  echo "   ($1: $(( $(date +%s) - t0 ))s)"
}
for a in coarse middle fine midfine all; do run $a $OUT/mask_$a.pt; done
cp $OUT/eval_shard.py.orig scripts/eval_shard.py && echo && echo "-- eval_shard.py restored to original"
echo; echo "-- 4. scoreboard (one count per story)"
$PY - <<'P'
import glob, pickle, re, random
random.seed(2)
DOGNAMES = ["Spot","Max","Buddy","Rex","Rover","Sparky","Fido","Bingo","Lucky","Fluffy","Benny","Toby","Charlie"]
DOG = r"\b(dog|dogs|puppy|puppies|pup)\b"
f = lambda p, T, fl=0: sum(bool(re.search(p, t, fl)) for t in T) / len(T) * 100
arms = ["none", "coarse", "middle", "fine", "midfine", "all"]
H = {a: [h for p in sorted(glob.glob(f"/data/lab/{'ncm3' if a=='none' else 'ncm7'}/{a}/shard_*.pkl")) for h in pickle.load(open(p, "rb"))["hyps"]] for a in arms}
print("arm      | Spot  | dog stories (word dog/puppy) | other dog names | Tim   | Lily  | top dog names")
for a, T in H.items():
    if not T: print(f"{a:<8} | no outputs"); continue
    others = [n for n in DOGNAMES if n != "Spot"]
    top = sorted(((sum(bool(re.search(rf"\b{n}\b", t)) for t in T), n) for n in DOGNAMES), reverse=True)[:4]
    sp, dg, ot = f(r"\bSpot\b", T), f(DOG, T, re.I), f(r"\b(" + "|".join(others) + r")\b", T)
    tm, ly = f(r"\bTim\b", T), f(r"\bLily\b", T); tops = dict((n, c) for c, n in top)
    print(f"{a:<8} | {sp:5.1f}% | {dg:5.1f}%                       | {ot:5.1f}%          | {tm:5.1f}% | {ly:5.1f}% | {tops}")
si = [i for i, t in enumerate(H["none"]) if re.search(r"\bSpot\b", t)]
print(f"\n{len(si)} normal stories mention Spot. Same slot, cooked with each lock (first 140 chars):")
for i in random.sample(si, min(3, len(si))):
    for a in ("none", "coarse", "middle", "midfine"):
        if H[a]: print(f"  [{a:<7}] {H[a][i][:140].replace(chr(10), ' ')}")
    print()
P
echo "genPPL (quality, lower = better) is the 'mean' line under each '-- arm:' above. Normal cook: 16.39 +/- 0.33"
echo "=== DONE. Tell Claude: ncm7 ready ==="
