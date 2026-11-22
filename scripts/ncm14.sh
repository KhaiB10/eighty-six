#!/usr/bin/env bash
# ncm14.sh (runs ON the Dell): eighty-six FINAL TEST, "same lock, three cook sizes". Real news (LM1B). One target, a real
# name: "obama" (7.3% of real 128-token news chunks). Audit its jars once on the SHARED pantry (2 rounds x 32 jars/shelf,
# the Lily recipe), then lock the SAME jars in the small (97M), medium (317M) and large (725M) cooks, plus a usage-matched
# RANDOM lock of the same size (placebo) in each.
# PREDICTION (written before running): the lock removes most "obama" at every size with no quality cost; the placebo
# changes nothing. Open question: does the bigger cook route around the lock more (weaker effect at larger size)?
# Also fixes the ncm13 counting bug: LM1B text is decoded from the raw token ids. Start with ncm14go.sh. ~35 min.
set -uo pipefail
OUT=/data/lab/ncm14; mkdir -p $OUT; exec > >(tee $OUT/REPORT.txt) 2>&1
cd /data/ncm/ncm; PY=/data/lab/venv_ncm/bin/python
echo "=== eighty-six FINAL: SAME LOCK, THREE COOK SIZES (obama) $(date) ==="
[ -f /data/ncm/audit/lm1b/valid_bert_30522.bin ] || { echo "!! need LM1B validation data (run ncm13.sh)"; exit 1; }
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



echo "-- 1. pantry audit on real news (shared LM1B pantry)"
$PY - <<'P'
import sys, os, random, numpy as np, torch
sys.path.insert(0, "/data/ncm/ncm"); os.chdir("/data/ncm/ncm")
from omegaconf import OmegaConf
from tokenization.vqvae_tokenizer import VQVAETokenizer
random.seed(0); torch.manual_seed(0)
CK = "artifacts/lm1b/small"; CFG = OmegaConf.load(f"{CK}/config.yaml"); tok = CFG.checkpoints.tokenizer
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(CK, tok) if os.path.exists(os.path.join(CK, tok)) else os.path.join("artifacts/lm1b", os.path.basename(tok))
tk = VQVAETokenizer.from_config(CFG.tokenizer); M = tk.vqvae.cuda().eval(); vq = M.vq; pt = tk.pre_tokenizer
LEVELS = [int(n) for n in vq.levels]; OFF = [0]
for n in LEVELS: OFF.append(OFF[-1] + n)
V = [int(v) for v in vq.vqvae_vocab_sizes]; T = int(CFG.tokenizer.vqvae.context_length)
PAD = pt.pad_token_id if pt.pad_token_id is not None else 0
TARGET = int(pt.encode("obama")[1]); print(f"   target 'obama' = token {TARGET} | shelves {LEVELS} | {V[0]} jars/shelf | chunk {T} tokens")
ids = np.fromfile("/data/ncm/audit/lm1b/valid_bert_30522.bin", dtype=np.uint16); off = np.load("/data/ncm/audit/lm1b/valid_bert_30522_offsets.npy")
rows_t, rows_b = [], []
for s in off.tolist():
    c = ids[s:s + T].astype(np.int64).tolist()
    if len(c) < T: c = c + [PAD] * (T - len(c))
    (rows_t if TARGET in c else rows_b).append(c)
random.shuffle(rows_t); random.shuffle(rows_b)
cut = int(len(rows_t) * 0.75); t_score, t_test, bg = rows_t[:cut][:1500], rows_t[cut:][:500], rows_b[:4000]
print(f"   chunks with obama: {len(rows_t)} (score {len(t_score)}, held-out {len(t_test)}) | background {len(bg)}")
@torch.no_grad()
def enc(R): return torch.cat([M.encode(torch.tensor(R[i:i+64], device="cuda")) for i in range(0, len(R), 64)])
@torch.no_grad()
def kept(R):
    x = torch.tensor(R, device="cuda"); idx = enc(R); out = []
    for i in range(0, len(R), 64):
        _, zl = vq.idx_to_zhats(idx[i:i+64], return_zlevels=True); out.append(M.decoder(zl[-1]).argmax(-1))
    y = torch.cat(out); real = x != PAD
    return (y == TARGET).any(1).float().mean().item(), ((y == x) & real).sum().item() / real.sum().item()
orig = [e.weight.data.clone() for e in vq.embeddings]
masked = {l: torch.zeros(V[l], dtype=torch.bool, device="cuda") for l in range(len(LEVELS))}
def apply():
    for l in range(len(LEVELS)):
        w = orig[l].clone(); w[masked[l]] = 1e4; vq.embeddings[l].weight.data.copy_(w)
xs = torch.tensor(t_score, device="cuda"); b_idx, p_idx = torch.nonzero(xs == TARGET, as_tuple=True)
k0, a0 = kept(t_test); print(f"   baseline pantry: obama rebuilt in {k0*100:.1f}% of held-out chunks | token accuracy {a0*100:.2f}%")
for r in (1, 2):
    apply(); si, bi = enc(t_score), enc(bg)
    for l in range(len(LEVELS)):
        codes = si[:, OFF[l]:OFF[l+1]][b_idx, (p_idx * LEVELS[l]) // T]
        cl = torch.bincount(codes, minlength=V[l]).float(); cb = torch.bincount(bi[:, OFF[l]:OFF[l+1]].reshape(-1), minlength=V[l]).float()
        sc = torch.log((cl + 0.5) / cl.sum()) - torch.log((cb + 0.5) / cb.sum()); sc[(cl < 3) | masked[l]] = -1e9
        top = torch.argsort(sc, descending=True)[:32]; masked[l][top[sc[top] > -1e8]] = True
    apply(); k, a = kept(t_test)
    print(f"   round {r}: {sum(int(m.sum()) for m in masked.values())} jars locked | pantry rebuilds obama in {k*100:.1f}% (was {k0*100:.1f}%) | accuracy {a*100:.2f}%")
for l in range(len(LEVELS)): vq.embeddings[l].weight.data.copy_(orig[l])
LOCK = {l: torch.nonzero(masked[l]).flatten().cpu() for l in masked}
# usage-matched random placebo (same shelves, same count, similar background use)
bi = enc(bg); RAND = {}
for l in range(len(LEVELS)):
    c = torch.bincount(bi[:, OFF[l]:OFF[l+1]].reshape(-1), minlength=V[l]).cpu(); order = torch.argsort(c).tolist(); rank = {j: i for i, j in enumerate(order)}
    tgt = set(LOCK[l].tolist()); ch = set()
    for j in LOCK[l].tolist():
        cand = [order[i] for i in range(max(0, rank[j] - 40), min(len(order), rank[j] + 41)) if order[i] not in tgt and order[i] not in ch]
        if cand: ch.add(random.choice(cand))
    RAND[l] = torch.tensor(sorted(ch), dtype=torch.long)
for name, M_ in (("lock", LOCK), ("rand", RAND)):
    for g, lv in (("midfine", range(4, 12)), ("all", range(0, 12))):
        sub = {l: (M_[l] if l in lv else M_[l][:0]) for l in M_}
        torch.save(sub, f"/data/lab/ncm14/{name}_{g}.pt")
print("   masks: lock/rand x midfine (L4-11) / all (L0-11):", {l: len(v) for l, v in LOCK.items()})
P
[ -f $OUT/lock_all.pt ] || { echo "!! audit failed"; exit 1; }
echo "-- 3. patch the cook (v2: also locks the first shelf) (guarded; inactive without NCM_MASK)"
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




run() {  # arm size mask
  local t0=$(date +%s); echo; echo "-- arm: $1"
  NCM_MASK=$3 timeout 7200 $PY evaluate.py --config config/eval/lm1b-$2.yaml --checkpoint-dir artifacts/lm1b/$2 --out $OUT/$1 2>&1 \
    | grep -vE "[Ww]arn|deprecat|Loading weights|it/s\]" | tail -1
  echo "   ($1: $(( ($(date +%s) - t0) / 60 )) min)"
}
for sz in small medium large; do
  run ${sz}_lock_all $sz $OUT/lock_all.pt
  run ${sz}_rand_all $sz $OUT/rand_all.pt
  run ${sz}_lock_midfine $sz $OUT/lock_midfine.pt
done
cp $OUT/eval_shard.py.orig scripts/eval_shard.py && echo && echo "-- eval_shard.py restored to original"
echo; echo "-- 4. scoreboard (text decoded from raw token ids; one count per generated 128-token chunk)"
$PY - <<'P'
import glob, pickle, re
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
POL = r"\b(bush|clinton|mccain|biden|palin|romney|pelosi|cheney|blair|cameron|sarkozy|putin|merkel|brown)\b"
def load(d):
    out = []
    for p in sorted(glob.glob(f"{d}/shard_*.pkl")):
        r = pickle.load(open(p, "rb")); src = r.get("raw_ids") or r.get("toks")
        out += [tok.decode([int(t) for t in x], skip_special_tokens=True) for x in src]
    return out
f = lambda p, T: sum(bool(re.search(p, t)) for t in T) / len(T) * 100
BASE = {"small": "/data/lab/ncm1/lm1b-small", "medium": "/data/lab/ncm13/lm1b-medium", "large": "/data/lab/ncm13/lm1b-large"}
print("size   | arm          | obama | president | other politicians | government | police | game/team")
for sz in ("small", "medium", "large"):
    for arm, d in (("normal", BASE[sz]), ("LOCK all", f"/data/lab/ncm14/{sz}_lock_all"), ("RANDOM all", f"/data/lab/ncm14/{sz}_rand_all"), ("LOCK mid+fine", f"/data/lab/ncm14/{sz}_lock_midfine")):
        T = load(d)
        if not T: print(f"{sz:<6} | {arm:<12} | no outputs"); continue
        o, pr, po, gv, pc, gm = f(r"\bobama\b", T), f(r"\bpresident\b", T), f(POL, T), f(r"\bgovernment\b", T), f(r"\bpolice\b", T), f(r"\b(game|team)\b", T)
        print(f"{sz:<6} | {arm:<12} | {o:5.1f}% | {pr:5.1f}%    | {po:5.1f}%            | {gv:5.1f}%     | {pc:5.1f}% | {gm:5.1f}%")
    print()
T = load("/data/lab/ncm14/large_lock_all"); B = load(BASE["large"])
oi = [i for i, t in enumerate(B) if re.search(r"\bobama\b", t)][:3]
print("Same slot, large cook, normal vs LOCK all (first 150 chars):")
for i in oi:
    print("  [normal]", B[i][:150]); print("  [LOCK  ]", T[i][:150] if i < len(T) else "-"); print()
P
echo "genPPL (quality, lower = better) is the 'mean' line under each '-- arm:'. Normal: small 93.15 | medium 88.36 | large 81.12"
echo "=== DONE. Tell Claude: ncm14 ready ==="
