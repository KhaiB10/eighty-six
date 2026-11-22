#!/usr/bin/env bash
# ncm6r.sh (runs ON the Dell): eighty-six PHASE 6 RECON, "can the pantry already spell a new name?"
# Before teaching NCM a new name (Khai) by adding jars, check: (1) Khai never appears in TinyStories; (2) how BPE splits
# it; (3) can the existing pantry already rebuild "Khai" stories (made by renaming Lily -> Khai)? (4) do Khai's middle-shelf
# jars differ from Lily's? (5) print the generator's training path (forward, inputs, heads) so the trainer is written
# against real code. ~2 min, light GPU, read-only.
set -uo pipefail
OUT=/data/lab/ncm6r; mkdir -p $OUT; exec > >(tee $OUT/REPORT.txt) 2>&1
cd /data/ncm/ncm; PY=/data/lab/venv_ncm/bin/python
echo "=== eighty-six PHASE 6 RECON $(date) ==="
$PY - <<'P'
import sys, os, re, random, collections, torch
sys.path.insert(0, "/data/ncm/ncm"); os.chdir("/data/ncm/ncm")
from omegaconf import OmegaConf
from tokenization.vqvae_tokenizer import VQVAETokenizer
random.seed(0)
CK = "artifacts/tinystories"; CFG = OmegaConf.load(f"{CK}/config.yaml")
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(CK, CFG.checkpoints.tokenizer)
CFG.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(CK, CFG.checkpoints.bpe_vocab)
CFG.tokenizer.pre_tokenizer.merges_file_path = os.path.join(CK, CFG.checkpoints.bpe_merges)
tk = VQVAETokenizer.from_config(CFG.tokenizer); M = tk.vqvae.cuda().eval(); vq = M.vq; pt = tk.pre_tokenizer
LEVELS = [int(n) for n in vq.levels]; OFF = [0]
for n in LEVELS: OFF.append(OFF[-1] + n)
T = 128; PAD = pt.pad_token_id if pt.pad_token_id is not None else pt.eos_token_id
txt = open("/data/ncm/audit/TinyStoriesV2-GPT4-valid.txt", encoding="utf-8", errors="ignore").read()
print("-- 1. is 'Khai' new?  occurrences in TinyStories validation:", len(re.findall(r"\bKhai\b", txt)))
for w in [" Khai", "Khai", " Lily"]:
    ids = pt.encode(w); print(f"-- 2. BPE split of {w!r}: {ids} -> {[pt.decode([i]) for i in ids]}")
st = [s.strip() for s in txt.split("<|endoftext|>") if "Lily" in s and len(s) > 200]; random.shuffle(st)
rows_l, rows_k = [], []
for s in st[:600]:
    a, b = pt.encode(s)[:T], pt.encode(s.replace("Lily", "Khai"))[:T]
    if len(a) < 48 or len(b) < 48: continue
    rows_l.append(a + [PAD]*(T-len(a))); rows_k.append(b + [PAD]*(T-len(b)))
rows_l, rows_k = rows_l[:500], rows_k[:500]
@torch.no_grad()
def enc(rows): return torch.cat([M.encode(torch.tensor(rows[i:i+64], device="cuda")) for i in range(0, len(rows), 64)])
@torch.no_grad()
def rec(idx):
    out = []
    for i in range(0, len(idx), 64):
        _, zl = vq.idx_to_zhats(idx[i:i+64], return_zlevels=True); out.append(M.decoder(zl[-1]).argmax(-1))
    return torch.cat(out)
xk = torch.tensor(rows_k, device="cuda"); ik = enc(rows_k); yk = rec(ik); real = xk != PAD
kept = sum("Khai" in pt.decode([t for t in y if t != PAD]) for y in yk.tolist()) / len(rows_k)
print(f"-- 3. existing pantry on 500 'Khai' stories: token accuracy {((yk==xk)&real).sum().item()/real.sum().item()*100:.2f}% | 'Khai' rebuilt in {kept*100:.1f}% of stories")
print("      (if ~100%, the pantry can already SPELL Khai with old jars; the job is teaching the COOK to choose them)")
il = enc(rows_l)
print("-- 4. per shelf: how often Khai stories use a different jar than the same Lily story (all positions | name positions)")
xl = torch.tensor(rows_l, device="cuda")
for l in range(len(LEVELS)):
    a, b = il[:, OFF[l]:OFF[l+1]], ik[:, OFF[l]:OFF[l+1]]
    print(f"      L{l:<2} ({LEVELS[l]:>3} slots): {((a!=b).float().mean()*100):5.1f}% of jars differ")
# how many brand-new jar combinations does Khai need at the middle shelves? (jars at Khai positions never seen at Lily positions)
P
echo; echo "-- 5. generator training path (for the trainer)"
sed -n '1,120p' models/next_concept.py | grep -n -E "def |self\.lm_heads|_SharedHeads|_TiedCodebookHead|tie_heads|max_vocab|vqvae_local|register_buffer|self\.levels|_lvl_first" | head -60
echo "##### NextConceptTransformer.__init__ (heads + levels)"
awk 'NR>=65 && NR<=312' models/next_concept.py | grep -n -E "lm_head|_Tied|_Shared|max_vocab|levels|_lvl_first|predict_level0|wte|wle|tie" | head -60
echo "##### forward + inputs"
sed -n '357,407p' models/next_concept.py
echo "##### quant.py: idx_to_ncm_input, get_next_autoregressive_input, idx_to_zhats"
sed -n '635,712p' models/quant.py
echo "##### eval_shard.py: how generation is set up"
sed -n '40,160p' scripts/eval_shard.py
echo "##### generator.pt contents"
$PY -c "
import torch; o=torch.load('artifacts/tinystories/generator.pt',map_location='cpu',weights_only=False)
print('top keys:', list(o.keys())[:12] if isinstance(o,dict) else type(o))
sd=o.get('model',o) if isinstance(o,dict) else o
ks=[k for k in sd if any(s in k for s in ('lm_head','E','wte','wle','l0'))]; print('\n'.join(f'  {k} {tuple(sd[k].shape)}' for k in ks[:40] if hasattr(sd[k],'shape')))
if isinstance(o,dict):
  for k,v in o.items():
    if k!='model' and not hasattr(v,'shape'): print('  meta',k,str(v)[:300])
"
echo "=== DONE. Tell Claude: ncm6r ready ==="
