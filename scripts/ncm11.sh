#!/usr/bin/env bash
# ncm11.sh (runs ON the Dell): eighty-six STEP 4, "predict, then test". Step 3's overlap rule makes predictions for names
# we have never locked. Lock each name's MIDDLE-shelf jars (L4-7; same recipe as Lily/Spot) and check:
#   Max (shares jars with dog names on the middle shelves) -> PREDICTION: Max drops AND dog stories drop, like Spot.
#   Sue (girl name, mostly its own jars there)              -> PREDICTION: Sue drops, girl stories stay, like Lily.
# Written BEFORE running. Start it with ncm11go.sh. ~15 min.
set -uo pipefail
OUT=/data/lab/ncm11; mkdir -p $OUT; exec > >(tee $OUT/REPORT.txt) 2>&1
cd /data/ncm/ncm; PY=/data/lab/venv_ncm/bin/python
echo "=== eighty-six STEP 4: PREDICT, THEN TEST (Max, Sue) $(date) ==="
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


for NAME in Max Sue; do
  echo "-- 1. pantry audit for $NAME (2 rounds, Lily recipe)"; mkdir -p $OUT/pantry_$NAME
  NAME=$NAME $PY - <<'Q'
import os
n = os.environ["NAME"]; s = open("/data/lab/ncm2b.sh").read()
for old, new in [('NAME, CONTROL = "Lily", "Tim"', f'NAME, CONTROL = "{n}", "Tim"'), ("K, ROUNDS, ROWS = 32, 8, []", "K, ROUNDS, ROWS = 32, 2, []"),
                 ('name_score, name_test = got["name"][:1500], got["name"][1500:]', 'cut = int(len(got["name"]) * 0.75); name_score, name_test = got["name"][:cut], got["name"][cut:]'),
                 ("/data/lab/ncm2b", f"/data/lab/ncm11/pantry_{n}")]:
    assert old in s, f"ncm2b layout changed: {old[:40]}"; s = s.replace(old, new)
s = s[:s.index('echo; echo "-- COOK RECON')] + "echo pantry-done\n"
open(f"/data/lab/ncm11/pantry_{n}/make.sh", "w").write(s)
Q
  bash $OUT/pantry_$NAME/make.sh > $OUT/pantry_$NAME/log.txt 2>&1
  grep -E "token ids that are|name split|^\s+[12] \|" $OUT/pantry_$NAME/log.txt | sed 's/^/   /'
  [ -f $OUT/pantry_$NAME/lily_mask.pt ] || { echo "!! pantry step failed for $NAME:"; tail -15 $OUT/pantry_$NAME/log.txt; exit 1; }
  NAME=$NAME $PY -c "
import torch, os; n = os.environ['NAME']; M = torch.load(f'/data/lab/ncm11/pantry_{n}/lily_mask.pt')
sub = {l: (v if 4 <= l <= 7 else v[:0]) for l, v in M.items()}; torch.save(sub, f'/data/lab/ncm11/mask_{n}_middle.pt')
print(f'   {n} middle lock: {sum(len(v) for v in sub.values())} jars')"
done
echo "-- 2. patch the cook (v2: also locks the first shelf) (guarded; inactive without NCM_MASK)"
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
            # v2 fix: the first shelf is drawn by eval_shard's unigram sampler (L0_P), outside the cook. Redraw any locked jar.
            if len(_MASK[0]):
                import sys as _s; _P0 = getattr(_s.modules["__main__"], "L0_P", None)
                _bad = _t.isin(idx, _MASK[0].to(idx.device))
                if _P0 is not None and _bad.any():
                    _p = _P0.clone().to(idx.device).float(); _p[_MASK[0].to(idx.device)] = 0; _p = _p / _p.sum()
                    idx = idx.clone(); idx[_bad] = _t.multinomial(_p, int(_bad.sum()), replacement=True)
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
run Max_middle $OUT/mask_Max_middle.pt
run Sue_middle $OUT/mask_Sue_middle.pt
cp $OUT/eval_shard.py.orig scripts/eval_shard.py && echo && echo "-- eval_shard.py restored to original"
echo; echo "-- 3. did the predictions hold? (one count per story)"
$PY - <<'P'
import glob, pickle, re
DOG = r"\b(dog|dogs|puppy|puppies|pup|pups|doggy|doggie|woof)\b"
GIRLS = r"\b(Lily|Lola|Anna|Amy|Rose|Sue|Lucy|Mia|Jill|Ellie|Sara|Sarah|Kitty|Princess)\b"
BOYS = r"\b(Tim|Tom|Ben|Sam|Jack|Bob|Timmy|Joe|Billy)\b"
def load(d): return [h for p in sorted(glob.glob(f"{d}/shard_*.pkl")) for h in pickle.load(open(p, "rb"))["hyps"]]
def f(p, T, fl=0): return sum(bool(re.search(p, t, fl)) for t in T) / len(T) * 100
ARMS = [("normal", "/data/lab/ncm3/none"), ("Max middle", "/data/lab/ncm11/Max_middle"), ("Sue middle", "/data/lab/ncm11/Sue_middle"),
        ("(ref) Spot middle", "/data/lab/ncm7/middle"), ("(ref) Lily middle", "/data/lab/ncm3d/middle")]
print("arm               | Max   | Sue   | Spot  | Lily  | dog stories | girl hero | boy names (no Max) | she/her")
for name, d in ARMS:
    T = load(d)
    if not T: print(f"{name:<17} | no outputs"); continue
    mx, su, sp, ly = f(r"\bMax\b", T), f(r"\bSue\b", T), f(r"\bSpot\b", T), f(r"\bLily\b", T)
    dg, gi, bo, sh = f(DOG, T, re.I), f(GIRLS, T), f(BOYS, T), f(r"\b(she|her)\b", T)
    print(f"{name:<17} | {mx:5.1f}% | {su:5.1f}% | {sp:5.1f}% | {ly:5.1f}% | {dg:5.1f}%      | {gi:5.1f}%    | {bo:5.1f}%             | {sh:5.1f}%")
print("\nPredictions: Max middle -> Max down AND dog stories down (like Spot). Sue middle -> Sue down, girl hero ~42% (like Lily).")
print("Margin of error ~ +/-2-3 points at 15-40%, ~ +/-1 point at ~5%.")
P
echo "genPPL (quality) per arm is the 'mean' line under each '-- arm:' above. Normal cook: 16.39 +/- 0.33"
echo "=== DONE. Tell Claude: ncm11 ready ==="
