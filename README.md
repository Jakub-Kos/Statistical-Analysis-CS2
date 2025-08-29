# Statistical Analysis of CS2 — Data Pipeline

Scripts for building a queryable dataset of Counter-Strike 2 matches for the course **NMAI059 – Pravděpodobnost a statistika 1 (MFF UK)**.

This repo contains a small ETL pipeline that:

1) **Fetches** HLTV demo archives using your session cookies  
2) **Parses** `.dem` files into structured data with `awpy`  
3) **Loads** per-match SQLite databases and **merges** them into a single warehouse

---

## Repository Structure

- `fetch_parse_delete.sh`  
  Uses your HLTV cookies to download demo archives (≈ **500–1200 MB per match**), extracts the `.dem` files, and hands them off to the parsing step. Cleans up intermediate files afterward.

- `parse_match_to_sqlite.py`  
  Parses `.dem` files with the **`awpy`** library and produces a **per-match `.sqlite`** database you can query.

- `data_warehouse.py`  
  Merges all per-match SQLite databases into a single **`warehouse.sqlite`** for easier analysis.

- `demo_urls.txt`  
  A text file with ~600 demo links.

- `data/warehouse.sqlite`  
  Example merged database (≈100 matches) to illustrate the final format.

---

## How It Works (Overview)

1. **Prepare links:** Put HLTV demo URLs into `demo_urls.txt`.
2. **Fetch & extract:** Run `fetch_parse_delete.sh` to download archives and extract `.dem` files (requires your HLTV cookies).
3. **Parse demos:** Run `parse_match_to_sqlite.py` to convert each `.dem` into a per-match SQLite DB using `awpy`.
4. **Build warehouse:** Run `data_warehouse.py` to merge all per-match DBs into a single `warehouse.sqlite`.

> ⚠️ **Storage & bandwidth:** Demo archives are large. Expect tens to hundreds of GB if you fetch many matches.

---

## Requirements

- Python (version commonly supported by `awpy`)
- `awpy` (for parsing demos)
- SQLite (CLI or any SQLite client)
- Bash (for the shell script)
- Your **HLTV cookies** (for authenticated demo downloads)

> Check each script’s header or `--help` (if available) for exact arguments and options.

---

## Quickstart

> The exact CLI flags may differ; see script comments/`--help` inside the repo.

```bash
# 1) Fetch archives & extract .dem files
bash fetch_parse_delete.sh

# 2) Parse .dem files into per-match SQLite DBs
python parse_match_to_sqlite.py

# 3) Merge per-match DBs into a single warehouse
python data_warehouse.py
```

Open the resulting `data/warehouse.sqlite` with your favorite SQLite tool and start querying.

---

## Notes & Caveats

- **HLTV terms of use:** You are responsible for how you access and use HLTV-provided files. Ensure your usage complies with their terms and any applicable laws.  
- **Cookies:** The fetch script relies on your HLTV session cookies to access downloads. Handle them carefully.
- **Performance:** Parsing and merging can take time on large batches; plan accordingly.

---

## Disclaimer

I don’t know the exact terms governing HLTV demo files. Use this code and any fetched data **at your own risk** and in compliance with HLTV’s rules and local regulations.

---

## Acknowledgements

- Demo parsing powered by the excellent **`awpy`** library.
- Data sources: **HLTV** demo archives.
