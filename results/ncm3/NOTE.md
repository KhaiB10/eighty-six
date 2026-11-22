**Known issue in this report's final counts:** the counter in `ncm3.sh` treated several fields of each generated sample
(the story text plus two token-id copies) as separate stories, so it reports ~3,842 "stories" instead of 1,280, and one
line printed a vocabulary dump by mistake. Every arm was counted the same way, so the comparison (Lily 14.9% → 0%) holds,
but use the corrected one-per-story counts in `results/ncm3c/REPORT.txt` and REPORT.md section C. genPPL numbers are unaffected.
