#!/usr/bin/env bash
# ncm6.sh (runs ON the Dell): eighty-six PHASE 6, "teach the cook a new name". The pantry can already spell Khai (95%),
# so the job is teaching the COOK to choose those jars. NCM has one shared brain plus 11 "station cooks" (output heads,
# one per shelf). Question: which station should learn a new name? Arms (150 steps each, 1,000 stories where Lily->Khai):
#   full = retrain the whole cook (classic fine-tune, the forgetting baseline)
#   heads_all / heads_top (L1-3) / heads_mid (L4-7) / heads_fine (L8-11) = train only those station cooks, brain frozen
# Measures: Khai learned (generation rate + held-out loss), forgetting (loss on normal stories), Lily / girls / quality
# (GPT-2 large perplexity, same scorer as the paper). This is the first TRAINING we run. ~20-25 min, full P40.
set -uo pipefail
OUT=/data/lab/ncm6; mkdir -p $OUT; exec > >(tee $OUT/REPORT.txt) 2>&1
cd /data/ncm/ncm; PY=/data/lab/venv_ncm/bin/python
echo "=== eighty-six PHASE 6: TEACH THE COOK A NEW NAME $(date) ==="
# v3: run 2 died at step 99 (Aster's Qwen reloaded to 15 GB and/or the ssh session closed). Now: pause Ollama for the
# run and restart it at the end (Aster is back afterwards), run DETACHED (see ncm6go.sh), station arms first, 150 steps.
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY - <<'P'
import sys, os, re, random, time, copy, math, json, torch, torch.nn.functional as F
sys.path.insert(0, "/data/ncm/ncm"); os.chdir("/data/ncm/ncm")
from omegaconf import OmegaConf
from tokenization.vqvae_tokenizer import VQVAETokenizer
from models.next_concept import NextConceptTransformer
random.seed(0); torch.manual_seed(0); t0 = time.time()
CK = "artifacts/tinystories"; CFG = OmegaConf.load(f"{CK}/config.yaml")
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(CK, CFG.checkpoints.tokenizer)
CFG.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(CK, CFG.checkpoints.bpe_vocab)
CFG.tokenizer.pre_tokenizer.merges_file_path = os.path.join(CK, CFG.checkpoints.bpe_merges)
tk = VQVAETokenizer.from_config(CFG.tokenizer); tk.vqvae = tk.vqvae.cuda().eval(); VQ = tk.vqvae; pt = tk.pre_tokenizer
for p in VQ.parameters(): p.requires_grad_(False)
net = NextConceptTransformer(vqvae_local=VQ, config=CFG.model).cuda()
sd = torch.load(os.path.join(CK, CFG.checkpoints.generator), map_location="cpu", weights_only=False)
net.load_state_dict(sd["model"], strict=True); BASE = copy.deepcopy(net.state_dict())
L0_P = sd["inference"]["level0_unigram"].float().cuda()
LEVELS = [int(n) for n in VQ.vq.levels]; T = 128
EOS = pt.eos_token_id; PAD = pt.pad_token_id if pt.pad_token_id is not None else EOS
print(f"cook loaded | shelves {LEVELS} | {len(net.lm_heads)} station cooks (heads for L1-L11) | {time.time()-t0:.0f}s")

# ---------- data ----------
txt = open("/data/ncm/audit/TinyStoriesV2-GPT4-valid.txt", encoding="utf-8", errors="ignore").read()
st = [s.strip() for s in txt.split("<|endoftext|>") if len(s.strip()) > 200]; random.shuffle(st)
def row(s):
    ids = pt.encode(s)[:T]; return ids + [PAD] * (T - len(ids)) if len(ids) >= 48 else None
khai, normal = [], []
for s in st:
    if "Lily" in s and len(khai) < 1200:
        r = row(s.replace("Lily", "Khai"))
        if r and "Khai" in pt.decode([t for t in r if t != PAD]): khai.append(r)
    elif "Lily" not in s and len(normal) < 500:
        r = row(s); normal.append(r) if r else None
    if len(khai) >= 1200 and len(normal) >= 500: break
k_train, k_test = khai[:1000], khai[1000:1200]
print(f"Khai stories: train {len(k_train)} / held-out {len(k_test)} | normal held-out {len(normal)}")

# ---------- teacher-forced loss (how surprised is the cook by a story) ----------
def batch_xy(rows):
    ids = torch.tensor(rows, device="cuda")
    x_ncm, y = VQ.get_ncm_inputs(ids)
    zp = VQ.encoder.wte(torch.full((len(rows), 1), EOS, device="cuda"))
    return torch.cat([zp, x_ncm], 1), y
def loss_of(rows, bs=50):
    net.eval(); tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            x, y = batch_xy(rows[i:i+bs]); lg = net(x)
            tot += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="sum").item(); n += y.numel()
    return tot / n

# ---------- generation + checks ----------
from transformers import AutoModelForCausalLM, AutoTokenizer
g2t = AutoTokenizer.from_pretrained("gpt2-large"); g2 = AutoModelForCausalLM.from_pretrained("gpt2-large").cuda().eval()
def gen(n=512, bs=128, seeds=(42, 0)):
    net.eval(); H = []
    for sd_ in seeds:
        torch.manual_seed(sd_)
        for s in range(0, n // len(seeds), bs):
            B = min(bs, n // len(seeds) - s)
            l0 = torch.multinomial(L0_P, B * LEVELS[0], replacement=True).view(B, LEVELS[0])
            with torch.no_grad():
                g = net.generate(l0, torch.full((B, 1), EOS, device="cuda"), temperature=1.0, top_k=0, top_p=1.0)
                _, zl = VQ.vq.idx_to_zhats(g, return_zlevels=True); ids = VQ.decoder(zl[-1]).argmax(-1).tolist()
            for r in ids:
                r = r[:r.index(EOS)] if EOS in r else r
                H.append(pt.decode([t for t in r if t not in (EOS, PAD)], skip_special_tokens=True))
    return H
def genppl(H):
    nll, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(H), 16):
            enc = [g2t(h)["input_ids"][:128] for h in H[i:i+16]]; enc = [e for e in enc if len(e) > 1]
            if not enc: continue
            L = max(map(len, enc)); ids = torch.tensor([e + [g2t.eos_token_id] * (L - len(e)) for e in enc], device="cuda")
            m = torch.tensor([[1] * len(e) + [0] * (L - len(e)) for e in enc], device="cuda")
            lg = g2(ids, attention_mask=m).logits[:, :-1]; tgt = ids[:, 1:]; mm = m[:, 1:].bool()
            nll += F.cross_entropy(lg[mm], tgt[mm], reduction="sum").item(); cnt += mm.sum().item()
    return math.exp(nll / cnt)
GIRLS = r"\b(Lily|Lola|Anna|Amy|Rose|Sue|Lucy|Mia|Jill|Ellie|Sara|Sarah|Kitty|Princess)\b"
def has(p, H): return sum(bool(re.search(p, h)) for h in H) / len(H) * 100

# ---------- arms ----------
HEADS = {"heads_all": range(0, 11), "heads_top": range(0, 3), "heads_mid": range(3, 7), "heads_fine": range(7, 11)}
ARMS = ["none", "heads_mid", "heads_top", "heads_fine", "heads_all", "full"]   # the real question first, slow baseline last
LR = {"full": 5e-5, "heads_all": 1e-3, "heads_top": 1e-3, "heads_mid": 1e-3, "heads_fine": 1e-3}
STEPS, BS, MB = 150, 32, 8
ROWS = []
print("\narm        | trained params | Khai in new stories | Khai loss (held-out) | normal-story loss (forgetting) | Lily | girl hero | genPPL | train s")
def run_arm(arm):
    net.load_state_dict(BASE); ts = time.time()
    for p in net.parameters(): p.requires_grad_(False)
    if arm == "full":
        for p in net.parameters(): p.requires_grad_(True)
    elif arm in HEADS:
        for i in HEADS[arm]:
            for p in net.lm_heads[i].parameters(): p.requires_grad_(True)
    params = [p for p in net.parameters() if p.requires_grad]; n_tr = sum(p.numel() for p in params)
    if params:
        opt = torch.optim.AdamW(params, lr=LR[arm], weight_decay=0.0); net.train()
        g2.cpu(); torch.cuda.empty_cache()
        for step in range(STEPS):
            opt.zero_grad(set_to_none=True); rows = random.sample(k_train, BS); tl = 0.0
            for j in range(0, BS, MB):   # small bites, same total batch of 32
                x, y = batch_xy(rows[j:j+MB]); lg = net(x)
                loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1)) * (MB / BS)
                loss.backward(); tl += loss.item(); del lg, loss
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            if step % 25 == 0 or step == STEPS - 1:
                print(f"   [{arm}] step {step:>3}/{STEPS} train loss {tl:.3f} | {(time.time()-ts)/(step+1):.1f}s per step", flush=True)
        if step == STEPS - 1: print(f"   [{arm}] peak GPU memory {torch.cuda.max_memory_allocated()/2**30:.1f} GB", flush=True)
        del opt; torch.cuda.empty_cache(); g2.cuda()
    tr_s = time.time() - ts
    lk, ln = loss_of(k_test), loss_of(normal); H = gen(); ppl = genppl(H)
    r = dict(arm=arm, params=n_tr, khai=has(r"\bKhai\b", H), khai_loss=lk, normal_loss=ln, lily=has(r"\bLily\b", H), girls=has(GIRLS, H), genppl=ppl, train_s=tr_s)
    ROWS.append(r)
    print(f"{arm:<10} | {n_tr/1e6:6.2f}M        | {r['khai']:5.1f}%              | {lk:.4f}             | {ln:.4f}                       | "
          f"{r['lily']:4.1f}% | {r['girls']:5.1f}%    | {ppl:6.2f} | {tr_s:4.0f}", flush=True)
    kh = [h for h in H if "Khai" in h][:1]
    if kh: print(f"   sample: {kh[0][:170]}")
for arm in ARMS:
    try: run_arm(arm)
    except Exception as e:
        print(f"!! arm {arm} failed: {type(e).__name__}: {str(e)[:200]}"); torch.cuda.empty_cache(); g2.cuda()
json.dump(ROWS, open("/data/lab/ncm6/results.json", "w"), indent=1)
b = ROWS[0]
print(f"\nHow to read: 'none' is the starting cook. Learning = Khai% up and Khai loss down. Forgetting = normal-story loss up"
      f" (start {b['normal_loss']:.4f}) and genPPL up (start {b['genppl']:.2f}; scored on 512 stories, so compare arms here, not to the paper's 16.39)."
      f" Lily% dropping is expected: Khai stories were made from Lily stories. | total {time.time()-t0:.0f}s")
P
echo "=== DONE. Tell Claude: ncm6 ready ==="
