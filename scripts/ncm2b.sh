#!/usr/bin/env bash
# ncm2a.sh (runs ON the Dell): NCM PHASE 2B, "keep pulling jars": iterative masking + cook recon.
# Question: does a specific detail (the name "Lily") live in a few specialty jars on certain shelves, or in common jars
# used by everything? Then: pull Lily's top jars from coarse / middle / fine shelves and see whether Lily disappears
# while the rest of each story survives. Uses only the released TinyStories tokenizer (the pantry), not the generator.
# ~5-15 min, light GPU. Read-only on the Dell except /data/lab/ncm2b.
set -uo pipefail
OUT=/data/lab/ncm2b; mkdir -p $OUT; exec > >(tee $OUT/REPORT.txt) 2>&1
echo "=== NCM PHASE 2B: KEEP PULLING JARS $(date) ==="
PY=/data/lab/venv_ncm/bin/python
$PY - <<'P'
import sys, os, re, json, math, random, time, collections, inspect
sys.path.insert(0, "/data/ncm/ncm"); os.chdir("/data/ncm/ncm")
import torch
from omegaconf import OmegaConf
from tokenization.vqvae_tokenizer import VQVAETokenizer
torch.manual_seed(0); random.seed(0); t0 = time.time()
OUT = "/data/lab/ncm2b"; NAME, CONTROL = "Lily", "Tim"

# ---------- load the pantry (same way scripts/eval_shard.py does) ----------
CK = "artifacts/tinystories"; CFG = OmegaConf.load(f"{CK}/config.yaml")
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(CK, CFG.checkpoints.tokenizer)
CFG.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(CK, CFG.checkpoints.bpe_vocab)
CFG.tokenizer.pre_tokenizer.merges_file_path = os.path.join(CK, CFG.checkpoints.bpe_merges)
tk = VQVAETokenizer.from_config(CFG.tokenizer); M = tk.vqvae.cuda().eval(); vq = M.vq
LEVELS = [int(n) for n in vq.levels]; OFF = [0]
for n in LEVELS: OFF.append(OFF[-1] + n)
T = int(CFG.tokenizer.vqvae.context_length); V = [int(v) for v in vq.vqvae_vocab_sizes]
pt = tk.pre_tokenizer; EOS = pt.eos_token_id; PAD = pt.pad_token_id if pt.pad_token_id is not None else EOS
print(f"shelves (positions per level): {LEVELS}  total {OFF[-1]} | jars per shelf: {V[0]} | dish length {T} tokens")
src = inspect.getsource(vq.z_to_idx)
print("z_to_idx uses:", sorted(set(re.findall(r"self\.(embeddings|ema_w_\w*|_sample_codes|_distances|stochastic\w*)", src))))

def ids_of(word):
    return {t for t in range(8192) if pt.decode([t]).strip() == word}
NAME_IDS, CTRL_IDS = ids_of(NAME), ids_of(CONTROL)
print(f"token ids that are exactly '{NAME}': {sorted(NAME_IDS)} | '{CONTROL}': {sorted(CTRL_IDS)}")
assert NAME_IDS, f"'{NAME}' is not a single token; audit needs a different name"

# ---------- test dishes ----------
txt = open("/data/ncm/audit/TinyStoriesV2-GPT4-valid.txt", encoding="utf-8", errors="ignore").read()
stories = [s.strip() for s in txt.split("<|endoftext|>") if len(s.strip()) > 200]; random.shuffle(stories)
def tok(s):
    ids = pt.encode(s)[:T]
    return ids + [PAD] * (T - len(ids)), len(ids)
want = dict(name=2000, bg=4000, retain=1000, ctrl=400); got = {k: [] for k in want}
for s in stories:
    if all(len(got[k]) >= want[k] for k in want): break
    has_n, has_c = NAME in s, CONTROL in s
    if has_n and len(got["name"]) < want["name"]:
        ids, n = tok(s)
        if n >= 48 and any(t in NAME_IDS for t in ids[:n]): got["name"].append(ids)
    elif not has_n:
        ids, n = tok(s)
        if n < 48: continue
        if has_c and len(got["ctrl"]) < want["ctrl"] and any(t in CTRL_IDS for t in ids[:n]): got["ctrl"].append(ids)
        elif len(got["bg"]) < want["bg"]: got["bg"].append(ids)
        elif len(got["retain"]) < want["retain"]: got["retain"].append(ids)
name_score, name_test = got["name"][:1500], got["name"][1500:]
print({k: len(v) for k, v in got.items()}, f"| name split: score {len(name_score)} / held-out test {len(name_test)} | {time.time()-t0:.0f}s")

# ---------- pantry in/out ----------
@torch.no_grad()
def encode(rows, bs=64):
    out = []
    for i in range(0, len(rows), bs):
        idx = M.encode(torch.tensor(rows[i:i+bs], device="cuda"))
        if isinstance(idx, (list, tuple)): idx = torch.cat([j.reshape(j.shape[0], -1) for j in idx], 1)
        out.append(idx)
    return torch.cat(out)
@torch.no_grad()
def recon(idx, bs=64):
    out = []
    for i in range(0, len(idx), bs):
        _, zl = vq.idx_to_zhats(idx[i:i+bs], return_zlevels=True)
        out.append(M.decoder(zl[-1]).argmax(-1))
    return torch.cat(out)
def level_codes(idx, l): return idx[:, OFF[l]:OFF[l+1]]
def pos_to_slot(l, p): return (p * LEVELS[l]) // T

TESTS = {"name": name_test, "retain": got["retain"], "ctrl": got["ctrl"]}
X = {k: torch.tensor(v, device="cuda") for k, v in TESTS.items()}
NAME_T = torch.tensor(sorted(NAME_IDS), device="cuda"); CTRL_T = torch.tensor(sorted(CTRL_IDS), device="cuda")
def metrics(idx_by_set):
    r = {}
    for k, idx in idx_by_set.items():
        x, y = X[k], recon(idx); real = x != PAD
        acc = ((y == x) & real).sum().item() / real.sum().item()
        if k == "name":
            isn = torch.isin(x, NAME_T)
            r["name_kept"] = torch.isin(y, NAME_T).any(1).float().mean().item()      # Lily anywhere in rebuilt story
            r["name_slot_kept"] = (torch.isin(y, NAME_T) & isn).sum().item() / isn.sum().item()
            rest = real & ~isn; r["rest_acc"] = ((y == x) & rest).sum().item() / rest.sum().item()
            r["_repl"] = collections.Counter(pt.decode([t]).strip() for t in y[isn].tolist()).most_common(6)
        elif k == "retain": r["retain_acc"] = acc
        else: r["ctrl_kept"] = torch.isin(y, CTRL_T).any(1).float().mean().item()
    return r

base_idx = {k: encode(v) for k, v in TESTS.items()}
assert base_idx["name"].shape[1] == OFF[-1], f"unexpected code layout {tuple(base_idx['name'].shape)}"
B = metrics(base_idx)
print(f"\n-- BASELINE (nothing pulled)  {time.time()-t0:.0f}s")
print(f"   {NAME} kept in rebuilt stories {B['name_kept']*100:.1f}% | {NAME} slots rebuilt exactly {B['name_slot_kept']*100:.1f}%")
print(f"   rest-of-story accuracy {B['rest_acc']*100:.2f}% | other stories accuracy {B['retain_acc']*100:.2f}% | {CONTROL} kept {B['ctrl_kept']*100:.1f}%")

# ---------- the audit: which jars sit under Lily, per shelf ----------
si, bi = encode(name_score), encode(got["bg"]); xs = torch.tensor(name_score, device="cuda")
b_idx, p_idx = torch.nonzero(torch.isin(xs, NAME_T), as_tuple=True)
print(f"\n-- AUDIT: {len(b_idx)} '{NAME}' positions in {len(name_score)} stories vs {len(got['bg'])} background stories")
print("   shelf slots | distinct jars under Lily | top-8 / top-32 / top-128 jars cover X% of Lily | those top-32 jars' share of ALL background use")
RANK, AUDIT = [], []
for l in range(len(LEVELS)):
    slots = pos_to_slot(l, p_idx); codes = level_codes(si, l)[b_idx, slots]
    cl = torch.bincount(codes, minlength=V[l]).float(); cb = torch.bincount(level_codes(bi, l).reshape(-1), minlength=V[l]).float()
    score = torch.log((cl + 0.5) / cl.sum()) - torch.log((cb + 0.5) / cb.sum()); score[cl < 3] = -1e9
    order = torch.argsort(score, descending=True); RANK.append(order)
    cov = [cl[order[:k]].sum().item() / cl.sum().item() for k in (8, 32, 128)]
    bgshare = cb[order[:32]].sum().item() / cb.sum().item()
    AUDIT.append(dict(level=l, slots=LEVELS[l], distinct=int((cl > 0).sum()), cov8=cov[0], cov32=cov[1], cov128=cov[2], bg_share32=bgshare))
    print(f"   L{l:<2} {LEVELS[l]:>4} | {int((cl>0).sum()):>5} | {cov[0]*100:5.1f} / {cov[1]*100:5.1f} / {cov[2]*100:5.1f} % | {bgshare*100:6.2f} %")

# ---------- keep pulling: pull Lily's top jars on every shelf, re-audit where she moved, pull again ----------
masked = {l: torch.zeros(V[l], dtype=torch.bool, device="cuda") for l in range(len(LEVELS))}
def apply_mask():
    for l in range(len(LEVELS)):
        w = orig[l].clone(); w[masked[l]] = 1e4; vq.embeddings[l].weight.data.copy_(w)
orig = [e.weight.data.clone() for e in vq.embeddings]
K, ROUNDS, ROWS = 32, 8, []
print(f"\n-- KEEP PULLING: each round pull the top {K} new Lily jars on all 12 shelves, then re-audit")
print("   round | jars pulled (% of pantry) | Lily GONE | rest-of-story acc | other stories acc | Tim kept | leaks | Lily became")
for r in range(1, ROUNDS + 1):
    apply_mask(); si, bi = encode(name_score), encode(got["bg"])
    for l in range(len(LEVELS)):
        codes = level_codes(si, l)[b_idx, pos_to_slot(l, p_idx)]
        cl = torch.bincount(codes, minlength=V[l]).float(); cb = torch.bincount(level_codes(bi, l).reshape(-1), minlength=V[l]).float()
        score = torch.log((cl + 0.5) / cl.sum()) - torch.log((cb + 0.5) / cb.sum()); score[(cl < 3) | masked[l]] = -1e9
        top = torch.argsort(score, descending=True)[:K]; masked[l][top[score[top] > -1e8]] = True
    apply_mask(); idx = {s: encode(v) for s, v in TESTS.items()}
    leaks = sum(torch.isin(level_codes(idx[s], l), torch.nonzero(masked[l]).flatten()).sum().item() for s in idx for l in range(len(LEVELS)))
    m = metrics(idx); n_m = sum(int(x.sum()) for x in masked.values()); gone = 1 - m["name_kept"]
    ROWS.append(dict(round=r, jars=n_m, gone=gone, rest_acc=m["rest_acc"], retain_acc=m["retain_acc"], ctrl_kept=m["ctrl_kept"], leaks=leaks, became=m["_repl"]))
    print(f"   {r:>5} | {n_m:>5} ({n_m/sum(V)*100:4.1f}%) | {gone*100:6.1f}%  | {m['rest_acc']*100:6.2f}% ({(m['rest_acc']-B['rest_acc'])*100:+.2f}) | "
          f"{m['retain_acc']*100:6.2f}% ({(m['retain_acc']-B['retain_acc'])*100:+.2f}) | {m['ctrl_kept']*100:5.1f}% | {leaks} | " + ", ".join(f"{w}:{c}" for w, c in m["_repl"][:4]))
    if gone >= 0.98: print("   (Lily essentially gone; stopping)"); break
for l in range(len(LEVELS)): vq.embeddings[l].weight.data.copy_(orig[l])
torch.save({l: torch.nonzero(masked[l]).flatten().cpu() for l in masked}, f"{OUT}/lily_mask.pt")
json.dump(dict(baseline={k: v for k, v in B.items() if not k.startswith("_")}, rounds=ROWS), open(f"{OUT}/results.json", "w"), indent=1, default=str)
print(f"\nwrote {OUT}/results.json + lily_mask.pt (the pulled jars, for the cook test) | total {time.time()-t0:.0f}s")
P
echo; echo "-- COOK RECON: how the generator picks jars (for the next step: forbid Lily's jars while it writes)"
sed -n '400,703p' /data/ncm/ncm/models/next_concept.py
grep -n -E "generate\(|level0_unigram|cfg_scale" /data/ncm/ncm/scripts/eval_shard.py
echo "=== DONE. Tell Claude: ncm2b ready ==="
