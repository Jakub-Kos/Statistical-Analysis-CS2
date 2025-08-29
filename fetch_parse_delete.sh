#!/usr/bin/env bash
set -euo pipefail

# Keep your working cookie header AS-IS (with "Cookie: ...")
COOKIE_HEADER="Cookie: YOUR COOKIE HERE"

IN_LIST="demo_urls.txt"
WORKDIR="work"
RAW_DIR="$WORKDIR/raw"
EXTRACT_DIR="$WORKDIR/extract"
MATCH_DIR="data/matches"

mkdir -p "$RAW_DIR" "$EXTRACT_DIR" "$MATCH_DIR" data

log() { echo "[$(date '+%H:%M:%S')] $*"; }

download_one() {
  local url="$1" dest="$2"
  log "Downloading $url"
  wget --progress=bar:force \
       --header="$COOKIE_HEADER" \
       --header="User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0 Safari/537.36" \
       --header="Referer: https://www.hltv.org/results" \
       --header="Accept-Language: en-US,en;q=0.9,cs;q=0.8" \
       -O "$dest" "$url"
}

is_probably_html() {
  local path="$1"
  [[ ! -s "$path" || $(stat -c%s "$path") -lt 50000 ]] && return 0
  local info; info="$(file -b "$path" || true)"
  echo "$info" | grep -qi 'HTML\|ASCII text'
}

detect_and_extract() {
  local path="$1" outdir="$2"
  mkdir -p "$outdir"
  local info; info="$(file -b "$path" || true)"

  if echo "$info" | grep -qi 'Zip archive'; then
    unzip -o "$path" -d "$outdir" >/dev/null
  elif echo "$info" | grep -Eqi 'RAR|7-zip'; then
    unar -o "$outdir" -f "$path" >/dev/null
  elif echo "$info" | grep -qi 'bzip2 compressed'; then
    cp "$path" "$outdir/packed.bz2"
    bunzip2 -f "$outdir/packed.bz2"
  else
    # maybe .dem already
    if echo "$info" | grep -qi 'Valve Source\|HL2\|Dem'; then
      cp "$path" "$outdir/$(basename "${path%.bin}.dem")"
    else
      cp "$path" "$outdir/$(basename "${path%.bin}.dem")"
    fi
  fi

  # recursively un-bzip nested .dem.bz2 if any
  find "$outdir" -type f -name "*.dem.bz2" -exec bunzip2 -f {} + 2>/dev/null || true
}

# ---- main ----
command -v unzip >/dev/null || { echo "Install unzip"; exit 1; }
command -v unar  >/dev/null || { echo "Install unar (sudo dnf install unar)"; exit 1; }
command -v file  >/dev/null || { echo "Install file"; exit 1; }

[[ -f "$IN_LIST" ]] || { echo "Input file $IN_LIST not found"; exit 1; }
log "Input file: $IN_LIST ( $(wc -l < "$IN_LIST") URLs )"

while IFS= read -r url; do
  [[ -z "$url" ]] && continue
  base="$(echo "$url" | sed 's#[^0-9A-Za-z._-]#_#g')"
  raw="$RAW_DIR/$base.bin"

  # 1) download
  if ! download_one "$url" "$raw"; then
    log "Download failed for $url"; continue
  fi
  if is_probably_html "$raw"; then
    log "Looks like HTML/blocked (403/expired cookie). Refresh COOKIE_HEADER."; rm -f "$raw"; continue
  fi

  # 2) extract
  rm -rf "$EXTRACT_DIR"/* 2>/dev/null || true
  detect_and_extract "$raw" "$EXTRACT_DIR" || { log "Extraction failed → skipping"; rm -f "$raw"; continue; }

  # 3) gather demos recursively
  # after extraction:
  mapfile -t DEMS < <(find "$EXTRACT_DIR" -type f \( -iname "*.dem" -o -iname "*.DEM" \))

  # match slug from extracted top-level dir or fallback to raw name
  match_slug="$(basename "${raw%.bin}")"
  top_dirs=$(find "$EXTRACT_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
  if [[ "$top_dirs" -eq 1 ]]; then
    only_dir="$(find "$EXTRACT_DIR" -mindepth 1 -maxdepth 1 -type d)"
    match_slug="$(basename "$only_dir" | sed 's#[^0-9A-Za-z._-]#_#g')"
  fi

  out_sqlite="data/matches/${match_slug}.sqlite"
  log "Building single-file match DB → $out_sqlite"
  python3 parse_match_to_sqlite.py --outfile "$out_sqlite" "${DEMS[@]}" || log "Parser error; continuing"


  # 6) cleanup
  rm -f "$raw"
  rm -rf "$EXTRACT_DIR"/*
  sleep 1
done < "$IN_LIST"

log "DONE. Per-match exports are in $MATCH_DIR/"
