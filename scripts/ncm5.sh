#!/usr/bin/env bash
# ncm5.sh (runs ON the Dell): eighty-six PHASE 5, "86 a whole topic". Can NCM forget a CONCEPT (beach stories), not just
# a name? A word ban needs a list of every beach word; the jar lock is learned from example stories.
# Prediction from Phase 3d: a topic is "what dish" -> it should live on the COARSE shelves (the opposite of a name).
# Scoreboard (ELM, Gandikota et al. NeurIPS 2025): innocence (beach gone), seamlessness (fluent, genPPL),
# specificity (other topics kept: park, girl heroes, she/her). ~15 min, full P40.
set -uo pipefail
OUT=/data/lab/ncm5; mkdir -p $OUT; exec > >(tee $OUT/REPORT.txt) 2>&1
cd /data/ncm/ncm; PY=/data/lab/venv_ncm/bin/python
echo "=== eighty-six PHASE 5: 86 A WHOLE TOPIC (beach) $(date) ==="
[ -d /data/lab/ncm3/none ] || { echo "!! need /data/lab/ncm3/none (from ncm3.sh)"; exit 1; }
echo "-- 1. audit: which jars do beach stories use, per shelf (story level, every position)"
$PY - <<'P'
import sys, os, re, random, torch
sys.path.insert(0, "/data/ncm/ncm"); os.chdir("/data/ncm/ncm")
from omegaconf import OmegaConf
from tokenization.vqvae_tokenizer import VQVAETokenizer
random.seed(0); torch.manual_seed(0)
CK = "artifacts/tinystories"; CFG = OmegaConf.load(f"{CK}/config.yaml")
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(CK, CFG.checkpoints.tokenizer)
CFG.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(CK, CFG.checkpoints.bpe_vocab)
CFG.tokenizer.pre_tokenizer.merges_file_path = os.path.join(CK, CFG.checkpoints.bpe_merges)
tk = VQVAETokenizer.from_config(CFG.tokenizer); M = tk.vqvae.cuda().eval(); vq = M.vq; pt = tk.pre_tokenizer
LEVELS = [int(n) for n in vq.levels]; OFF = [0]
for n in LEVELS: OFF.append(OFF[-1] + n)
V = [int(v) for v in vq.vqvae_vocab_sizes]; T = 128; PAD = pt.pad_token_id if pt.pad_token_id is not None else pt.eos_token_id
BEACH = r"\b(beach|beaches|sand|sandy|sandcastle|sandcastles|ocean|sea|seaside|shore|shell|shells|seashell|seashells|wave|waves|surf|tide)\b"
ban = sorted({t for t in range(8192) if re.fullmatch(BEACH, pt.decode([t]).strip().lower() or "x")})
open("/data/lab/ncm5/ban_ids.txt", "w").write(",".join(map(str, ban)))
print("   word-ban list (beach words that are single tokens):", [pt.decode([t]).strip() for t in ban])
txt = open("/data/ncm/audit/TinyStoriesV2-GPT4-valid.txt", encoding="utf-8", errors="ignore").read()
st = [s.strip() for s in txt.split("<|endoftext|>") if len(s.strip()) > 200]; random.shuffle(st)
beach, bg = [], []
for s in st:
    if len(beach) >= 1500 and len(bg) >= 4000: break
    ids = pt.encode(s)[:T]
    if len(ids) < 48: continue
    window = pt.decode(ids)
    ids = ids + [PAD] * (T - len(ids))
    if re.search(BEACH, window, re.I):
        if len(beach) < 1500: beach.append(ids)
    elif not re.search(BEACH, s, re.I) and len(bg) < 4000: bg.append(ids)
print(f"   beach stories {len(beach)} | background {len(bg)}")
@torch.no_grad()
def enc(rows):
    return torch.cat([M.encode(torch.tensor(rows[i:i+64], device="cuda")) for i in range(0, len(rows), 64)])
eb, ebg = enc(beach), enc(bg)
print("   shelf | top-32 beach jars: share of beach-story use | share of background use | ratio")
RANK = []
for l in range(len(LEVELS)):
    cb = torch.bincount(eb[:, OFF[l]:OFF[l+1]].reshape(-1), minlength=V[l]).float()
    cg = torch.bincount(ebg[:, OFF[l]:OFF[l+1]].reshape(-1), minlength=V[l]).float()
    score = torch.log((cb + 0.5) / cb.sum()) - torch.log((cg + 0.5) / cg.sum()); score[cb < 5] = -1e9
    o = torch.argsort(score, descending=True); RANK.append(o)
    sb, sg = cb[o[:32]].sum() / cb.sum(), cg[o[:32]].sum() / cg.sum()
    print(f"   L{l:<2}  | {sb*100:6.2f}% | {sg*100:6.2f}% | {sb/max(sg,1e-9):6.1f}x")
GROUPS = {"coarse": range(0, 4), "middle": range(4, 8), "fine": range(8, 12), "all": range(0, 12)}
for g, lv in GROUPS.items():
    for k in (32, 128):
        m = {l: (RANK[l][:k].cpu() if l in lv else RANK[l][:0].cpu()) for l in range(len(LEVELS))}
        torch.save(m, f"/data/lab/ncm5/mask_{g}{k}.pt")
print("   masks written: coarse/middle/fine/all x 32/128 jars per shelf")
P
echo "-- 2. patch the cook (both patches guarded; inactive unless NCM_MASK / NCM_BAN is set)"
cp scripts/eval_shard.py $OUT/eval_shard.py.orig
ln=$(grep -n "^net = NextConceptTransformer(" scripts/eval_shard.py | head -1 | cut -d: -f1)
[ -z "$ln" ] && { echo "!! could not find where the cook is built"; exit 1; }
if ! grep -q "NCM_MASK_PATCH" scripts/eval_shard.py; then
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
run() {  # arm env
  local t0=$(date +%s); echo; echo "-- arm: $1"
  env $2 timeout 3600 $PY evaluate.py --config config/eval/tinystories.yaml --checkpoint-dir artifacts/tinystories --out $OUT/$1 2>&1 \
    | grep -vE "[Ww]arn|deprecat|Loading weights|it/s\]" | tail -1
  echo "   ($1: $(( $(date +%s) - t0 ))s)"
}
run ban "NCM_BAN=$(cat $OUT/ban_ids.txt)"
for a in coarse32 middle32 fine32 all32 coarse128; do run $a "NCM_MASK=$OUT/mask_$a.pt"; done
cp $OUT/eval_shard.py.orig scripts/eval_shard.py && echo && echo "-- eval_shard.py restored to original"
echo; echo "-- 3. scoreboard (one count per story)"
$PY - <<'P'
import glob, pickle, re, random
random.seed(1)
BEACH = r"\b(beach|beaches|sand|sandy|sandcastle|sandcastles|ocean|sea|seaside|shore|shell|shells|seashell|seashells|wave|waves|surf|tide)\b"
WATER = r"\b(water|swim|swimming|boat|fish|lake|river|pool)\b"
PARK, SHE = r"\bpark\b", r"\b(she|her)\b"
GIRLS = r"\b(Lily|Lola|Anna|Amy|Rose|Sue|Lucy|Mia|Jill|Ellie|Sara|Sarah|Kitty|Princess)\b"
f = lambda p, T: sum(bool(re.search(p, t, re.I if p in (BEACH, WATER) else 0)) for t in T) / len(T) * 100
arms = ["none", "ban", "coarse32", "middle32", "fine32", "all32", "coarse128"]
H = {a: [h for p in sorted(glob.glob(f"/data/lab/{'ncm3' if a=='none' else 'ncm5'}/{a}/shard_*.pkl")) for h in pickle.load(open(p, "rb"))["hyps"]] for a in arms}
print("arm       | BEACH (innocence: lower=better) | water/swim/boat | park | girl hero | she/her")
for a, T in H.items():
    if not T: print(f"{a:<9} | no outputs"); continue
    print(f"{a:<9} | {f(BEACH,T):5.1f}%                          | {f(WATER,T):5.1f}%          | {f(PARK,T):5.1f}% | {f(GIRLS,T):5.1f}%    | {f(SHE,T):5.1f}%")
bi = [i for i, t in enumerate(H["none"]) if re.search(BEACH, t, re.I)]
print(f"\n{len(bi)} normal stories mention the beach. Same slot, cooked with each lock (first 150 chars):")
for i in random.sample(bi, min(3, len(bi))):
    for a in ("none", "ban", "coarse32", "all32"):
        if H[a]: print(f"  [{a:<8}] {H[a][i][:150].replace(chr(10), ' ')}")
    print()
P
echo "genPPL (seamlessness, lower=better) is the 'mean' line under each '-- arm:' above. Normal cook: 16.39 +/- 0.33"
echo "=== DONE. Tell Claude: ncm5 ready ==="
