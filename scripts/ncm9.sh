#!/usr/bin/env bash
# ncm9.sh (runs ON the Dell): eighty-six STEP 2b, "the placebo". Every lock so far used jars picked FOR a target. Does a
# RANDOM lock of the same size, on the same shelves, cause the same changes (girl->boy, dog->rabbit)? If yes, part of
# the shelf story is just "locking the top shelves shakes up the plan". Random jars are matched to Lily's r2 jars shelf
# by shelf in COUNT and in how often they are used (so they are not unfairly common or rare). Also: the first-shelf leak
# is now fixed in the patch, and dog stories are recounted without the word "bark" (tree bark). Waits for ncm8 to finish.
# PREDICTION (written before running): random top-shelf locks change stories somewhat (more boys, fewer girls/dogs) but
# much less than targeted ones; random middle/fine locks change almost nothing. ~12 min.
set -uo pipefail
OUT=/data/lab/ncm9; mkdir -p $OUT; exec > >(tee $OUT/REPORT.txt) 2>&1
cd /data/ncm/ncm; PY=/data/lab/venv_ncm/bin/python
echo "=== eighty-six STEP 2b: PLACEBO $(date) ==="
while pgrep -f "ncm8.sh" >/dev/null; do echo "   waiting for ncm8 to finish... $(date +%H:%M)"; sleep 60; done
[ -f /data/lab/ncm3/r2/lily_mask.pt ] || { echo "!! need /data/lab/ncm3/r2/lily_mask.pt"; exit 1; }
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


echo "-- 1. build random locks matched to Lily's r2 jars (same shelves, same count, similar usage)"
$PY - <<'P'
import sys, os, random, torch
sys.path.insert(0, "/data/ncm/ncm"); os.chdir("/data/ncm/ncm")
from omegaconf import OmegaConf
from tokenization.vqvae_tokenizer import VQVAETokenizer
random.seed(7); torch.manual_seed(7)
CK = "artifacts/tinystories"; CFG = OmegaConf.load(f"{CK}/config.yaml")
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(CK, CFG.checkpoints.tokenizer)
CFG.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(CK, CFG.checkpoints.bpe_vocab)
CFG.tokenizer.pre_tokenizer.merges_file_path = os.path.join(CK, CFG.checkpoints.bpe_merges)
tk = VQVAETokenizer.from_config(CFG.tokenizer); M = tk.vqvae.cuda().eval(); vq = M.vq; pt = tk.pre_tokenizer
LEVELS = [int(n) for n in vq.levels]; OFF = [0]
for n in LEVELS: OFF.append(OFF[-1] + n)
V = [int(v) for v in vq.vqvae_vocab_sizes]; T = 128; PAD = pt.pad_token_id if pt.pad_token_id is not None else pt.eos_token_id
txt = open("/data/ncm/audit/TinyStoriesV2-GPT4-valid.txt", encoding="utf-8", errors="ignore").read()
st = [s.strip() for s in txt.split("<|endoftext|>") if len(s.strip()) > 200 and "Lily" not in s]; random.shuffle(st)
rows = []
for s in st:
    ids = pt.encode(s)[:T]
    if len(ids) >= 48: rows.append(ids + [PAD] * (T - len(ids)))
    if len(rows) >= 3000: break
with torch.no_grad():
    idx = torch.cat([M.encode(torch.tensor(rows[i:i+64], device="cuda")) for i in range(0, len(rows), 64)])
LILY = torch.load("/data/lab/ncm3/r2/lily_mask.pt")
RAND = {}
print("   shelf | Lily jars | their median background use | random jars' median use")
for l in range(len(LEVELS)):
    c = torch.bincount(idx[:, OFF[l]:OFF[l+1]].reshape(-1), minlength=V[l]).cpu()
    tgt = set(LILY[l].tolist()); order = torch.argsort(c).tolist(); rank = {j: r for r, j in enumerate(order)}
    chosen = set()
    for j in LILY[l].tolist():
        r0 = rank[j]; cand = [order[r] for r in range(max(0, r0 - 40), min(len(order), r0 + 41)) if order[r] not in tgt and order[r] not in chosen]
        if cand: chosen.add(random.choice(cand))
    RAND[l] = torch.tensor(sorted(chosen), dtype=torch.long)
    if len(tgt): print(f"   L{l:<2}  | {len(tgt):>4}      | {c[LILY[l]].float().median():8.0f}                     | {c[RAND[l]].float().median():8.0f}")
for name, lv in {"rand_coarse": range(0, 4), "rand_middle": range(4, 8), "rand_fine": range(8, 12), "rand_all": range(0, 12)}.items():
    sub = {l: (RAND[l] if l in lv else RAND[l][:0]) for l in RAND}
    torch.save(sub, f"/data/lab/ncm9/mask_{name}.pt"); print(f"   {name:<11} {sum(len(v) for v in sub.values())} random jars")
P
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
  echo "   ($1: $(( $(date +%s) - t0 ))s)"; grep -h "NCM_MASK level-0" $OUT/$1/shard_*.log 2>/dev/null | sed 's/^/   /'
}
for a in rand_coarse rand_middle rand_fine rand_all; do run $a $OUT/mask_$a.pt; done
cp $OUT/eval_shard.py.orig scripts/eval_shard.py && echo && echo "-- eval_shard.py restored to original"
echo; echo "-- 3. placebo vs targeted (one count per story; 'bark' no longer counts as a dog word)"
$PY - <<'P'
import glob, pickle, re, os
DOG = r"\b(dog|dogs|puppy|puppies|pup|pups|doggy|doggie|woof)\b"
GIRLS = r"\b(Lily|Lola|Anna|Amy|Rose|Sue|Lucy|Mia|Jill|Ellie|Sara|Sarah|Kitty|Princess)\b"
BOYS = r"\b(Tim|Tom|Ben|Sam|Max|Jack|Bob|Timmy|Joe|Billy)\b"
def load(d): return [h for p in sorted(glob.glob(f"{d}/shard_*.pkl")) for h in pickle.load(open(p, "rb"))["hyps"]]
def f(p, T, fl=0): return sum(bool(re.search(p, t, fl)) for t in T) / len(T) * 100
ARMS = [("normal", "/data/lab/ncm3/none"),
        ("RANDOM top L0-3", "/data/lab/ncm9/rand_coarse"), ("Lily top L0-3", "/data/lab/ncm3d/coarse"), ("Spot top L0-3", "/data/lab/ncm7/coarse"),
        ("RANDOM middle", "/data/lab/ncm9/rand_middle"), ("Lily middle", "/data/lab/ncm3d/middle"), ("Spot middle", "/data/lab/ncm7/middle"),
        ("RANDOM bottom", "/data/lab/ncm9/rand_fine"), ("Lily bottom", "/data/lab/ncm3d/fine"),
        ("RANDOM all", "/data/lab/ncm9/rand_all"), ("Lily all (r2)", "/data/lab/ncm3/r2"), ("Spot all", "/data/lab/ncm7/all")]
ARMS += [(f"dog-topic {a}", f"/data/lab/ncm8/{a}") for a in ("ban", "coarse32", "middle32", "fine32", "all32", "coarse128") if os.path.isdir(f"/data/lab/ncm8/{a}")]
print("arm               | Lily  | Spot  | girl hero | boy names | dog stories | she/her")
for name, d in ARMS:
    T = load(d)
    if not T: print(f"{name:<17} | no outputs at {d}"); continue
    ly, sp, gi, bo, dg, sh = f(r"\bLily\b", T), f(r"\bSpot\b", T), f(GIRLS, T), f(BOYS, T), f(DOG, T, re.I), f(r"\b(she|her)\b", T)
    print(f"{name:<17} | {ly:5.1f}% | {sp:5.1f}% | {gi:5.1f}%    | {bo:5.1f}%    | {dg:5.1f}%      | {sh:5.1f}%")
print("\nHow to read: compare each RANDOM row with the targeted rows on the same shelves. If RANDOM top also drops girls")
print("and dogs, the top-shelf effect is partly 'any lock up there shakes the plan'. Margin of error ~ +/-2-3 points at 15-40%.")
P
echo "genPPL (quality) per new arm is the 'mean' line under each '-- arm:' above. Normal cook: 16.39 +/- 0.33"
echo "=== DONE. Tell Claude: ncm9 ready ==="
