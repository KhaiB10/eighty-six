#!/usr/bin/env bash
# ncm11b.sh (runs ON the Dell): recount for step 4, ~20 s, no GPU. The Sue lock lowered "girl hero" (a 14-name list) but
# not she/her. Were the girls still there under names the list misses (the pantry test turned Sue into Jen, Sally...)?
# Count every capitalised name that follows "named"/"called", and girl stories by she/her + a much larger name list.
/data/lab/venv_ncm/bin/python - <<'P'
import glob, pickle, re, collections
def load(d): return [h for p in sorted(glob.glob(f"{d}/shard_*.pkl")) for h in pickle.load(open(p, "rb"))["hyps"]]
A = {"normal": "/data/lab/ncm3/none", "Sue middle": "/data/lab/ncm11/Sue_middle", "Lily middle": "/data/lab/ncm3d/middle", "Max middle": "/data/lab/ncm11/Max_middle"}
H = {a: load(d) for a, d in A.items()}
named = lambda T: collections.Counter(m for t in T for m in set(re.findall(r"\b(?:named|called)\s+([A-Z][a-z]+)", t)))
base = named(H["normal"])
for a in ("Sue middle", "Max middle"):
    c = named(H[a]); gain = sorted(((c[n] - base[n], n) for n in set(c) | set(base)), reverse=True)
    print(f"{a}: names that GAINED the most vs normal (stories):", ", ".join(f"{n} +{g}" for g, n in gain[:10] if g > 0))
    print(f"{a}: names that LOST the most:", ", ".join(f"{n} {g}" for g, n in sorted(gain)[:5] if g < 0))
GIRLS2 = r"\b(Lily|Lola|Anna|Amy|Rose|Sue|Lucy|Mia|Jill|Ellie|Sara|Sarah|Kitty|Princess|Jen|Jenny|Sally|Daisy|Mimi|Luna|Molly|Emma|Lisa|Kim|Jane|Mary|Katie|Kate|Ann|Ella|Zoe|Meg|Pam|Beth|Amy|Lilly|Rosie|Bella|Ruby|Polly|Susie|Sophie|Grace|Nina|Lila|Ivy|Hannah)\b"
f = lambda p, T, fl=0: sum(bool(re.search(p, t, fl)) for t in T) / len(T) * 100
print("\narm         | girl hero (BIG name list) | she/her stories | 'a little girl'/'a girl' | Sue   | Lily")
for a, T in H.items():
    g, s, lg, su, ly = f(GIRLS2, T), f(r"\b(she|her)\b", T), f(r"\b(little girl|a girl|the girl)\b", T, re.I), f(r"\bSue\b", T), f(r"\bLily\b", T)
    print(f"{a:<11} | {g:5.1f}%                    | {s:5.1f}%          | {lg:5.1f}%                   | {su:5.1f}% | {ly:5.1f}%")
P
