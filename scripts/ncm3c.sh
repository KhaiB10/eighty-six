#!/usr/bin/env bash
# ncm3c.sh (runs ON the Dell): NCM 3C, recount only (~20 s, no GPU). Uses the 'hyps' field (the final story text, one per
# sample) for every arm, and shows a few stories from the word-ban vs the jar-lock arms so we can read how each one copes.
OUT=/data/lab/ncm3c; mkdir -p $OUT; exec > >(tee $OUT/REPORT.txt) 2>&1
/data/lab/venv_ncm/bin/python - <<'P'
import glob, pickle, re, collections, random
random.seed(0)
GIRLS = ["Lola","Anna","Amy","Rose","Sue","Lucy","Mia","Jill","Ellie","Sara","Kitty"]
w = lambda x, t: re.search(rf"\b{x}\b", t) is not None
ARMS = [("none", "/data/lab/ncm3"), ("r2", "/data/lab/ncm3"), ("r8", "/data/lab/ncm3"), ("ban", "/data/lab/ncm3b")]
H = {}
for arm, base in ARMS:
    H[arm] = [h for f in sorted(glob.glob(f"{base}/{arm}/shard_*.pkl")) for h in pickle.load(open(f, "rb"))["hyps"]]
print("arm  | stories | Lily  | Tim   | other girls | any girl incl. Lily | top names")
for arm, T in H.items():
    n = len(T); g = collections.Counter(x for t in T for x in GIRLS + ["Lily"] if w(x, t))
    print(f"{arm:<4} | {n:>7} | {sum(w('Lily',t) for t in T)/n*100:5.1f}% | {sum(w('Tim',t) for t in T)/n*100:5.1f}% | "
          f"{sum(any(w(x,t) for x in GIRLS) for t in T)/n*100:5.1f}%      | {sum(any(w(x,t) for x in GIRLS+['Lily']) for t in T)/n*100:5.1f}%              | {dict(g.most_common(5))}")
# same seed + same sample index = same starting jar, so these are the closest thing to "the same story, cooked 3 ways"
lily_i = [i for i, t in enumerate(H["none"]) if w("Lily", t)]
print(f"\n{len(lily_i)} normal-cook stories had Lily. Same sample slot, cooked 3 ways (first 160 chars):")
for i in random.sample(lily_i, min(4, len(lily_i))):
    for arm in ("none", "r2", "ban"):
        print(f"  [{arm:<4}] {H[arm][i][:160].replace(chr(10), ' ')}")
    print()
P
echo "=== DONE. Tell Claude: ncm3c ready ==="
