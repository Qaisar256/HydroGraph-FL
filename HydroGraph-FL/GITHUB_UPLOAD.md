# Upload to GitHub

**Target:** https://github.com/Qaisar256/HydroGraph-FL

1. Unzip the archive and open a terminal inside the extracted `HydroGraph-FL_GitHub_Ready` directory.
2. Install and verify: `python -m pip install -r requirements.txt` followed by `python -m unittest discover -s tests -v`.
3. For a quick check: `python run_neural.py --mode central --quick`.
4. Optional real-source download: `python scripts/download_field_data.py --start 2019-01-01 --end 2025-12-31` (requires internet access).
5. If this is an existing repository, copy the extracted content into your local clone and carefully reconcile any changed files, then commit:

```bash
git status
git add .
git commit -m "Add HydroGraph-FL GRU-GNN, federated/QDE pipeline and field-data downloader"
git push origin main
```

If starting a new local repository, run `git init`, `git branch -M main`, and `git remote add origin https://github.com/Qaisar256/HydroGraph-FL.git` before the above commit/push commands.

**Research record:** Published manuscript methods should disclose that bundled benchmark CSVs are simulation-produced, while observed WQP–USGS downloads are a distinct, not-yet-evaluated validation pathway. Before public release, confirm repository licensing and authorship with all contributors.
