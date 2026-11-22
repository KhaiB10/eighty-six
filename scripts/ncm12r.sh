#!/usr/bin/env bash
# ncm12r.sh (runs ON the Dell): eighty-six SCALE-UP RECON. Get the bigger kitchens ready: download Stanford's LM1B medium
# (317M) and large (725M) cooks (they share one pantry with LM1B small) and the OpenWebText model, print their configs,
# find real-world text for the audits, and check disk space. Downloads only, light GPU; Aster can stay on. ~10-25 min
# depending on download speed. Read-only on everything else.
set -uo pipefail
OUT=/data/lab/ncm12r; mkdir -p $OUT; exec > >(tee $OUT/REPORT.txt) 2>&1
cd /data/ncm/ncm; PY=/data/lab/venv_ncm/bin/python
echo "=== eighty-six SCALE-UP RECON $(date) ==="
echo "-- 0. disk space"; df -h /data | tail -1 | sed 's/^/   /'
echo "-- 1. what Stanford published (file sizes)"
$PY - <<'P'
from huggingface_hub import HfApi
api = HfApi()
for repo, kind in [("hazyresearch/ncm-checkpoints", "model"), ("hazyresearch/ncm-datasets", "dataset")]:
    try:
        info = api.model_info(repo, files_metadata=True) if kind == "model" else api.dataset_info(repo, files_metadata=True)
        tot = {}
        for f in info.siblings:
            top = "/".join(f.rfilename.split("/")[:2]) if f.rfilename.count("/") >= 2 else f.rfilename.split("/")[0]
            tot[top] = tot.get(top, 0) + (f.size or 0)
        print(f"   {repo}:")
        for k, v in sorted(tot.items()): print(f"      {k:<28} {v/1e9:6.2f} GB")
    except Exception as e: print(f"   !! {repo}: {type(e).__name__} {str(e)[:200]}")
P
echo "-- 2. download LM1B medium + large + OpenWebText (skips what is already here)"
$PY - <<'P'
from huggingface_hub import snapshot_download
for pats in (["lm1b/medium/*", "lm1b/large/*"], ["owt/*", "openwebtext/*"]):
    try:
        p = snapshot_download("hazyresearch/ncm-checkpoints", allow_patterns=pats, local_dir="artifacts"); print("   ok:", pats)
    except Exception as e: print("   !! download failed:", pats, type(e).__name__, str(e)[:200])
P
find artifacts -maxdepth 3 -name "*.pt" -newer /data/lab/ncm12r -o -maxdepth 3 -name "config.yaml" | grep -E "lm1b|owt|openweb" | sort | sed 's/^/   /'
du -sh artifacts/lm1b/* artifacts/owt* artifacts/openweb* 2>/dev/null | sed 's/^/   /'
echo "-- 3. configs (levels, codebooks, text tokenizer, generator size, generation settings)"
for c in artifacts/lm1b/small/config.yaml artifacts/lm1b/medium/config.yaml artifacts/lm1b/large/config.yaml $(ls artifacts/owt*/config.yaml artifacts/openweb*/config.yaml 2>/dev/null); do
  [ -f $c ] || continue; echo "##### $c"
  $PY - "$c" <<'P'
import sys; from omegaconf import OmegaConf
C = OmegaConf.load(sys.argv[1]); m, t = C.model, C.tokenizer
print("   generator: n_layer", m.get("n_layer"), "n_embd", m.get("n_embd"), "n_head", m.get("n_head"), "| context", m.get("context_length"), "| prefix_len", m.get("prefix_len"))
print("   pre_tokenizer:", dict(t.pre_tokenizer))
vq = t.vqvae; print("   pantry: embed_dim", vq.get("embed_dim"), "| context", vq.get("context_length"), "| codes/level", list(vq.vqvae_vocab_size)[:3], "... x", len(vq.vqvae_vocab_size))
lv = vq.vector_quantizer_config.levels; print("   levels:", list(lv.manual_levels) if lv.get("use_manual_levels") else dict(lv))
print("   generation:", dict(C.get("generation", {})), "| eval:", {k: C.eval.get(k) for k in ("n_samples", "seeds", "batch_size", "ref_model")})
P
done
echo "-- 4. real-world text for the audits"
$PY - <<'P'
from huggingface_hub import HfApi
api = HfApi()
try:
    fs = [f for f in api.list_repo_files("hazyresearch/ncm-datasets", repo_type="dataset") if "lm1b" in f.lower() or "owt" in f.lower() or "openweb" in f.lower()]
    print("   ncm-datasets lm1b/owt files:", fs[:20] or "none")
except Exception as e: print("   !! listing ncm-datasets:", e)
for repo in ["dvruette/lm1b", "lm1b", "Skylion007/openwebtext", "stas/openwebtext-10k"]:
    try:
        fs = api.list_repo_files(repo, repo_type="dataset"); print(f"   {repo}: {len(fs)} files, e.g. {fs[:4]}")
    except Exception as e: print(f"   {repo}: not reachable ({type(e).__name__})")
P
echo "-- 5. does the LM1B pantry load, and how are names tokenized? (uncased tokenizers lowercase everything)"
$PY - <<'P'
import sys, os; sys.path.insert(0, "/data/ncm/ncm"); os.chdir("/data/ncm/ncm")
from omegaconf import OmegaConf
from tokenization.vqvae_tokenizer import VQVAETokenizer
CK = "artifacts/lm1b/small"; CFG = OmegaConf.load(f"{CK}/config.yaml")
tok = CFG.checkpoints.tokenizer; tp = os.path.join(CK, tok) if os.path.exists(os.path.join(CK, tok)) else os.path.join("artifacts/lm1b", os.path.basename(tok))
CFG.tokenizer.vqvae.checkpoint_path = tp
try:
    tk = VQVAETokenizer.from_config(CFG.tokenizer); pt = tk.pre_tokenizer
    for w in ["obama", "Obama", "paris", "china", "london", "football", "iraq", "microsoft"]:
        ids = pt.encode(w); print(f"   {w!r:<12} -> {ids} -> {[pt.decode([i]) for i in ids]}")
except Exception as e: print("   !! pantry load failed:", type(e).__name__, str(e)[:300])
P
echo "=== DONE. Tell Claude: ncm12r ready ==="
