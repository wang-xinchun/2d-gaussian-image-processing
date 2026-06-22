#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <type> <command> <result> [status] [workdir]" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
TYPE="$1"
COMMAND_TEXT="$2"
RESULT_TEXT="$3"
STATUS_TEXT="${4:-success}"
WORKDIR_TEXT="${5:-$ROOT_DIR}"
TS_ISO="$(TZ=Asia/Shanghai date '+%Y-%m-%dT%H:%M:%S%z')"
TS_HUMAN="$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S CST')"

mkdir -p "$LOG_DIR"
touch "$LOG_DIR/operations.jsonl" "$LOG_DIR/operations.md"

LOCK_DIR="$LOG_DIR/.append_operation.lock"
while ! mkdir "$LOCK_DIR" 2>/dev/null; do
  sleep 0.05
done
trap 'rmdir "$LOCK_DIR"' EXIT

JSONL_MAX_ID="$(python3 - "$LOG_DIR/operations.jsonl" <<'PY'
from pathlib import Path
import re
import sys
path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8") if path.exists() else ""
ids = [int(x) for x in re.findall(r'"operation_id":"OP-(\d{6})"', text)]
print(max(ids) if ids else 0)
PY
)"
MD_MAX_ID="$(python3 - "$LOG_DIR/operations.md" <<'PY'
from pathlib import Path
import re
import sys
path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8") if path.exists() else ""
ids = [int(x) for x in re.findall(r'OP-(\d{6})', text)]
print(max(ids) if ids else 0)
PY
)"
if [ "$JSONL_MAX_ID" -ge "$MD_MAX_ID" ]; then
  NEXT_OP_INDEX=$((JSONL_MAX_ID + 1))
else
  NEXT_OP_INDEX=$((MD_MAX_ID + 1))
fi
OPERATION_ID="$(printf 'OP-%06d' "$NEXT_OP_INDEX")"

json_escape() {
  python3 -c 'import json,sys; print(json.dumps(sys.argv[1], ensure_ascii=False))' "$1"
}

TYPE_JSON="$(json_escape "$TYPE")"
COMMAND_JSON="$(json_escape "$COMMAND_TEXT")"
RESULT_JSON="$(json_escape "$RESULT_TEXT")"
STATUS_JSON="$(json_escape "$STATUS_TEXT")"
WORKDIR_JSON="$(json_escape "$WORKDIR_TEXT")"

printf '{"operation_id":"%s","timestamp_bj_iso":"%s","timestamp_bj":"%s","actor":"assistant","type":%s,"workdir":%s,"command":%s,"status":%s,"result":%s}\n' \
  "$OPERATION_ID" "$TS_ISO" "$TS_HUMAN" "$TYPE_JSON" "$WORKDIR_JSON" "$COMMAND_JSON" "$STATUS_JSON" "$RESULT_JSON" >> "$LOG_DIR/operations.jsonl"

{
  printf '\n## %s | %s | %s\n\n' "$OPERATION_ID" "$TS_HUMAN" "$STATUS_TEXT"
  printf -- '- Actor: assistant\n'
  printf -- '- Type: %s\n' "$TYPE"
  printf -- '- Workdir: `%s`\n' "$WORKDIR_TEXT"
  printf -- '- Command: `%s`\n' "$COMMAND_TEXT"
  printf -- '- Result: %s\n' "$RESULT_TEXT"
} >> "$LOG_DIR/operations.md"

echo "$OPERATION_ID"
