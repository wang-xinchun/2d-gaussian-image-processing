#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <role> <content> [summary] [turn_id]" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
ROLE="$1"
CONTENT="$2"
SUMMARY="${3:-}"
TURN_ID_INPUT="${4:-}"
TS_ISO="$(TZ=Asia/Shanghai date '+%Y-%m-%dT%H:%M:%S%z')"
TS_HUMAN="$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S CST')"

mkdir -p "$LOG_DIR"
touch "$LOG_DIR/messages.jsonl" "$LOG_DIR/messages.md"

LOCK_DIR="$LOG_DIR/.append_message.lock"
while ! mkdir "$LOCK_DIR" 2>/dev/null; do
  sleep 0.05
done
trap 'rmdir "$LOCK_DIR"' EXIT

JSONL_MAX_ID="$(python3 - "$LOG_DIR/messages.jsonl" <<'PY'
from pathlib import Path
import re
import sys
path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8") if path.exists() else ""
ids = [int(x) for x in re.findall(r'"message_id":"MSG-(\d{6})"', text)]
print(max(ids) if ids else 0)
PY
)"
MD_MAX_ID="$(python3 - "$LOG_DIR/messages.md" <<'PY'
from pathlib import Path
import re
import sys
path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8") if path.exists() else ""
ids = [int(x) for x in re.findall(r'MSG-(\d{6})', text)]
print(max(ids) if ids else 0)
PY
)"
if [ "$JSONL_MAX_ID" -ge "$MD_MAX_ID" ]; then
  NEXT_MESSAGE_INDEX=$((JSONL_MAX_ID + 1))
else
  NEXT_MESSAGE_INDEX=$((MD_MAX_ID + 1))
fi
MESSAGE_ID="$(printf 'MSG-%06d' "$NEXT_MESSAGE_INDEX")"

if [ -n "$TURN_ID_INPUT" ]; then
  TURN_ID="$TURN_ID_INPUT"
else
  TURN_ID="$(printf 'T%04d' "$NEXT_MESSAGE_INDEX")"
fi

json_escape() {
  python3 -c 'import json,sys; print(json.dumps(sys.argv[1], ensure_ascii=False))' "$1"
}

SUMMARY_JSON="$(json_escape "$SUMMARY")"
CONTENT_JSON="$(json_escape "$CONTENT")"

printf '{"message_id":"%s","turn_id":"%s","timestamp_bj_iso":"%s","timestamp_bj":"%s","role":"%s","summary":%s,"content":%s}\n' \
  "$MESSAGE_ID" "$TURN_ID" "$TS_ISO" "$TS_HUMAN" "$ROLE" "$SUMMARY_JSON" "$CONTENT_JSON" >> "$LOG_DIR/messages.jsonl"

{
  LAST_TURN_HEADER="$(awk '/^## T[0-9]+/{value=$0} END{print value}' "$LOG_DIR/messages.md")"
  if [ "$LAST_TURN_HEADER" != "## $TURN_ID" ]; then
    printf '\n## %s\n\n' "$TURN_ID"
  fi
  printf '\n### %s | %s | %s\n\n' "$MESSAGE_ID" "$ROLE" "$TS_HUMAN"
  if [ -n "$SUMMARY" ]; then
    printf -- '- Summary: %s\n' "$SUMMARY"
  fi
  printf -- '- Content:\n\n```text\n%s\n```\n' "$CONTENT"
} >> "$LOG_DIR/messages.md"

echo "$MESSAGE_ID"
