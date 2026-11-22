#!/usr/bin/env bash
# ncm10.sh (runs ON the Dell): eighty-six STEP 3, "do names share jars with their siblings?" Pantry only, no story
# writing, ~3 min, light GPU (Aster can stay on). For each name, collect the jars used at that name's positions on every
# shelf, and compare jar lists between names (histogram overlap: 0 = no jars in common, 1 = identical use).
# Fairness: every name gets the same number of positions (subsampled); each name is also compared with itself
# (split in two halves) as the "same" ceiling.
# PREDICTION (written before running): on the middle shelves (L4-7) Spot shares MORE jars with other dog names than Lily
# does with other girl names. Max (dog AND boy name) overlaps with both dogs and boys.
set -uo pipefail
OUT=/data/lab/ncm10; mkdir -p $OUT; exec > >(tee $OUT/REPORT.txt) 2>&1
cd /data/ncm/ncm; PY=/data/lab/venv_ncm/bin/python
echo "=== eighty-six STEP 3: DO NAMES SHARE JARS? $(date) ==="
$PY - <<'P'
import sys, os, re, random, collections, torch
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
GROUPS = {"girl": ["Lily", "Sue", "Lucy", "Mia", "Amy", "Anna"], "boy": ["Tim", "Tom", "Ben", "Sam"], "dog": ["Spot", "Max", "Buddy", "Rex"]}
TYPE = {n: g for g, ns in GROUPS.items() for n in ns}
def ids_of(w): return {t for t in range(8192) if pt.decode([t]).strip() == w}
IDS = {n: ids_of(n) for n in TYPE}
for n, s in IDS.items():
    if not s: print(f"   (skipping {n}: not a single token)")
NAMES = [n for n in TYPE if IDS[n]]
txt = open("/data/ncm/audit/TinyStoriesV2-GPT4-valid.txt", encoding="utf-8", errors="ignore").read()
st = [s.strip() for s in txt.split("<|endoftext|>") if len(s.strip()) > 200]; random.shuffle(st)
DOGW = re.compile(r"\b(dog|dogs|puppy|puppies|pup|woof)\b", re.I)
rows = {n: [] for n in NAMES}; maxdog = [0, 0]
for s in st:
    for n in NAMES:
        if len(rows[n]) < 700 and re.search(rf"\b{n}\b", s):
            ids = pt.encode(s)[:T]
            if len(ids) >= 48 and any(t in IDS[n] for t in ids):
                rows[n].append(ids + [PAD] * (T - len(ids)))
                if n == "Max": maxdog[0] += 1; maxdog[1] += bool(DOGW.search(s))
print("   stories per name:", {n: len(r) for n, r in rows.items()})
print(f"   Max: {maxdog[1]}/{maxdog[0]} of its stories mention a dog -> Max is a dog in ~{maxdog[1]/max(maxdog[0],1)*100:.0f}% of stories")
NPOS = 1200   # same number of name positions for every name
H = {}
with torch.no_grad():
    for n in NAMES:
        R = rows[n]
        if not R: continue
        x = torch.tensor(R, device="cuda"); idx = torch.cat([M.encode(x[i:i+64]) for i in range(0, len(R), 64)])
        b, p = torch.nonzero(torch.isin(x, torch.tensor(sorted(IDS[n]), device="cuda")), as_tuple=True)
        perm = torch.randperm(len(b), device="cuda")[:NPOS]; b, p = b[perm], p[perm]
        half = len(b) // 2; H[n] = {"n": len(b)}
        for l in range(len(LEVELS)):
            codes = idx[:, OFF[l]:OFF[l+1]][b, (p * LEVELS[l]) // T]
            H[n][l] = torch.bincount(codes, minlength=V[l]).float()
            H[n][(l, "a")] = torch.bincount(codes[:half], minlength=V[l]).float()
            H[n][(l, "b")] = torch.bincount(codes[half:], minlength=V[l]).float()
NAMES = [n for n in NAMES if n in H and H[n]["n"] >= 300]
def ov(a, b): return torch.minimum(a / a.sum(), b / b.sum()).sum().item()
SH = {"top L0-3": range(0, 4), "middle L4-7": range(4, 8), "bottom L8-11": range(8, 12)}
def avg(n, m, lv): return sum(ov(H[n][l], H[m][l]) for l in lv) / len(lv)
def self_ov(n, lv): return sum(ov(H[n][(l, "a")], H[n][(l, "b")]) for l in lv) / len(lv)
print("\n-- overlap of jar use (0 = nothing in common, 1 = identical). 'self' = the name vs itself in two halves (the ceiling)")
for shelf, lv in SH.items():
    print(f"\n   {shelf}:  name   | type | positions | self  | with SIBLINGS (same type) | with OTHER types | siblings - other")
    for n in NAMES:
        sib = [m for m in NAMES if m != n and TYPE[m] == TYPE[n]]; oth = [m for m in NAMES if TYPE[m] != TYPE[n]]
        s_ = sum(avg(n, m, lv) for m in sib) / max(len(sib), 1); o_ = sum(avg(n, m, lv) for m in oth) / max(len(oth), 1)
        print(f"              {n:<6} | {TYPE[n]:<4} | {H[n]['n']:>5}     | {self_ov(n, lv):.2f}  | {s_:.2f}                      | {o_:.2f}             | {s_-o_:+.2f}")
print("\n-- Max, the dog-and-boy name: overlap with each group, per shelf group")
if "Max" in NAMES:
    for shelf, lv in SH.items():
        g = {t: sum(avg("Max", m, lv) for m in NAMES if TYPE[m] == t and m != "Max") / max(1, sum(1 for m in NAMES if TYPE[m] == t and m != "Max")) for t in GROUPS}
        print(f"   {shelf:<12}: with girls {g['girl']:.2f} | with boys {g['boy']:.2f} | with dogs {g['dog']:.2f}")
print("\n-- per-shelf profile, Lily vs Spot: overlap with own siblings, shelf by shelf")
for n in ("Lily", "Spot"):
    if n not in NAMES: continue
    sib = [m for m in NAMES if m != n and TYPE[m] == TYPE[n]]
    print(f"   {n:<5}: " + " ".join(f"L{l}:{sum(ov(H[n][l], H[m][l]) for m in sib)/len(sib):.2f}" for l in range(len(LEVELS))))
print("\nHow to read: high 'with SIBLINGS' on a shelf = the name shares jars with other names of its kind there, so locking")
print("it there also locks the kind. Compare Lily vs Spot on the middle shelves. 'siblings - other' > 0 = jars encode the kind.")
P
echo "=== DONE. Tell Claude: ncm10 ready ==="
