#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST_INPUT="${1:-papers/top_venues/manifest.tsv}"

if [[ "$MANIFEST_INPUT" = /* ]]; then
  MANIFEST="$MANIFEST_INPUT"
else
  MANIFEST="$ROOT_DIR/$MANIFEST_INPUT"
fi

PACK_DIR="$(dirname "$MANIFEST")"
CHECKSUM_FILE="$PACK_DIR/checksums.sha256"

if [ ! -f "$MANIFEST" ]; then
  echo "Missing manifest: $MANIFEST" >&2
  exit 1
fi

declare -a downloaded_files=()

download_with_python() {
  local url="$1"
  local output_path="$2"

  python3 - "$url" "$output_path" <<'PY'
import sys
import time

import requests

url = sys.argv[1]
output_path = sys.argv[2]
headers = {"User-Agent": "Mozilla/5.0"}

last_error = None
for attempt in range(1, 6):
    try:
        with requests.get(url, headers=headers, timeout=600, stream=True) as response:
            response.raise_for_status()
            with open(output_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        break
    except Exception as exc:
        last_error = exc
        time.sleep(attempt)
else:
    raise SystemExit(f"download failed: {last_error!r}")
PY
}

while IFS=$'\t' read -r category venue year short_name title pdf_url official_source note local_rel_path; do
  if [ -z "${category:-}" ]; then
    continue
  fi

  case "$category" in
    \#*) continue ;;
  esac

  dest_path="$ROOT_DIR/$local_rel_path"
  dest_dir="$(dirname "$dest_path")"
  mkdir -p "$dest_dir"

  if [ -s "$dest_path" ]; then
    echo "Skipping existing [$venue $year] $short_name"
    downloaded_files+=("$dest_path")
    continue
  fi

  temp_path="${dest_path}.part"
  echo "Downloading [$venue $year] $short_name"
  rm -f "$temp_path"
  download_with_python "$pdf_url" "$temp_path"
  if [ ! -s "$temp_path" ]; then
    echo "Download produced no file: $temp_path" >&2
    exit 1
  fi
  mv "$temp_path" "$dest_path"
  downloaded_files+=("$dest_path")
done < "$MANIFEST"

if [ "${#downloaded_files[@]}" -eq 0 ]; then
  echo "No files were downloaded." >&2
  exit 1
fi

{
  for file_path in "${downloaded_files[@]}"; do
    sha256sum "$file_path"
  done
} > "$CHECKSUM_FILE"

echo "Saved checksums to $CHECKSUM_FILE"
