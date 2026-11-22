**Known error in this report:** the scoreboard at the end of `REPORT.txt` read the beach-test folder (`ncm5`) instead of
this run's own outputs, because of a missed rename when the script was copied. The generations and genPPL numbers in
this run are correct. The corrected dog-topic counts are in `results/ncm9/REPORT.txt` (bottom rows) and in REPORT.md,
section I. `scripts/ncm8.sh` has been fixed.
