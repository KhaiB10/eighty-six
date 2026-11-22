**Known issue in this report's target counts:** on LM1B the decoded sentences came back empty (each generated sample
starts with the end-of-text marker, so the text was cut to nothing), which is why every cook shows 0.0% for every word.
The genPPL replication numbers (medium 88.36, large 81.12) are correct. `ncm14.sh` decodes from the raw token ids instead;
see `results/ncm14/REPORT.txt` for the real counts.
