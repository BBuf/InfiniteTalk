#!/usr/bin/env bash
set -euo pipefail

# "Ultimate" client examples for:
# - Ulysses=8 + TeaCache
# - Video-to-Video (V2V) and Image-to-Video (I2V)
# - 480P and 720P
#
# This script will:
# - create 4 tasks (V2V-480 / V2V-720 / I2V-480 / I2V-720)
# - poll status until completion
# - download resulting mp4 files
#
# Usage:
#   bash serving/client_ulysses8_teacache_ultimate.sh
#
# Required inputs (set via env vars if your paths differ):
#   REF_VIDEO: conditioning video (for V2V)
#   REF_IMAGE: conditioning image (for I2V)
#   AUDIO_WAV: driving audio (.wav)
#
# Defaults match the README examples; adjust as needed.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

REF_VIDEO="${REF_VIDEO:-${ROOT}/examples/single/ref_video.mp4}"
REF_IMAGE="${REF_IMAGE:-${ROOT}/examples/single/ref_image.png}"
AUDIO_WAV="${AUDIO_WAV:-${ROOT}/examples/single/1.wav}"

PROMPT="${PROMPT:-a person is talking}"
INFER_STEPS="${INFER_STEPS:-40}"
MAX_FRAME_NUM="${MAX_FRAME_NUM:-250}"
SEED="${SEED:-42}"

USE_TEACACHE="${USE_TEACACHE:-true}"
TEACACHE_THRESH="${TEACACHE_THRESH:-0.2}"

OUT_DIR="${OUT_DIR:-${ROOT}/serving/demo_results}"
mkdir -p "${OUT_DIR}"

need_file() {
  local p="$1"
  local name="$2"
  if [[ ! -f "${p}" ]]; then
    echo "[ERROR] ${name} not found: ${p}" >&2
    echo "        Please export ${name} to a valid path, e.g.:" >&2
    echo "          export ${name}=/path/to/file" >&2
    exit 2
  fi
}

need_file "${AUDIO_WAV}" "AUDIO_WAV"
need_file "${REF_VIDEO}" "REF_VIDEO"
need_file "${REF_IMAGE}" "REF_IMAGE"

json_get() {
  local key="$1"
  python3 - "$key" <<'PY'
import json, sys

key = sys.argv[1]
raw = sys.stdin.read()
if not raw.strip():
  print(f"[ERROR] empty response body while extracting key={key}", file=sys.stderr)
  raise SystemExit(3)
try:
  data = json.loads(raw)
except Exception as e:
  print(f"[ERROR] response is not valid JSON while extracting key={key}: {e}", file=sys.stderr)
  print("[ERROR] raw response:", file=sys.stderr)
  print(raw, file=sys.stderr)
  raise SystemExit(3)

val = data.get(key)
if val is None:
  print(f"[ERROR] JSON key not found: {key}", file=sys.stderr)
  print(json.dumps(data, ensure_ascii=False, indent=2), file=sys.stderr)
  raise SystemExit(3)
print(val)
PY
}

create_task() {
  local tag="$1"
  local cond_path="$2"
  local size="$3"
  local shift="$4"
  local save_name="$5"

  local payload
  payload="$(python3 - "${cond_path}" "${AUDIO_WAV}" "${size}" "${shift}" "${save_name}" <<'PY'
import json, os, sys
cond_path = sys.argv[1]
audio_path = sys.argv[2]
size = sys.argv[3]
shift = float(sys.argv[4])
save_name = sys.argv[5]

payload = {
  "prompt": os.environ.get("PROMPT", "a person is talking"),
  "image_path": cond_path,
  "audio_path": audio_path,
  "infer_steps": int(os.environ.get("INFER_STEPS", "40")),
  "max_frame_num": int(os.environ.get("MAX_FRAME_NUM", "250")),
  "seed": int(os.environ.get("SEED", "42")),
  "size": size,
  "sample_shift": shift,
  "use_teacache": (os.environ.get("USE_TEACACHE", "true").lower() in ("1","true","yes","y","on")),
  "teacache_thresh": float(os.environ.get("TEACACHE_THRESH", "0.2")),
  "save_result_path": save_name,
}
print(json.dumps(payload, ensure_ascii=False))
PY
  )"

  echo "[INFO] Create task: ${tag}" >&2

  # Capture body + HTTP status code (last line).
  local raw
  raw="$(curl -sS -X POST "${BASE_URL}/v1/tasks/video" \
    -H "Content-Type: application/json" \
    -d "${payload}" \
    -w $'\n%{http_code}')"

  local http_code body
  http_code="${raw##*$'\n'}"
  body="${raw%$'\n'*}"

  if [[ -z "${http_code}" || "${http_code}" == "${raw}" ]]; then
    echo "[ERROR] Failed to parse HTTP code from response." >&2
    echo "[ERROR] Raw response:" >&2
    echo "${raw}" >&2
    exit 3
  fi

  if [[ "${http_code}" != "200" ]]; then
    echo "[ERROR] Create task failed: HTTP ${http_code}" >&2
    echo "[ERROR] Response body:" >&2
    echo "${body}" >&2
    exit 3
  fi

  local task_id
  task_id="$(json_get "task_id" <<<"${body}")"
  echo "[INFO]   task_id=${task_id}" >&2

  # IMPORTANT: only echo task_id to stdout (so command substitution stays clean).
  echo "${task_id}"
}

poll_and_download() {
  local tag="$1"
  local task_id="$2"
  local out_mp4="$3"

  echo "[INFO] Polling: ${tag} (${task_id})"
  while true; do
    local raw http_code status_json st
    raw="$(curl -sS "${BASE_URL}/v1/tasks/${task_id}/status" -w $'\n%{http_code}')"
    http_code="${raw##*$'\n'}"
    status_json="${raw%$'\n'*}"

    if [[ "${http_code}" != "200" ]]; then
      echo "[ERROR] Poll status failed for ${task_id}: HTTP ${http_code}" >&2
      echo "[ERROR] Response body:" >&2
      echo "${status_json}" >&2
      return 4
    fi

    st="$(json_get "status" <<<"${status_json}")"
    if [[ "${st}" == "completed" ]]; then
      echo "[INFO]   completed."
      break
    fi
    if [[ "${st}" == "failed" || "${st}" == "cancelled" ]]; then
      echo "[ERROR]   task ended with status=${st}" >&2
      echo "[ERROR]   status response: ${status_json}" >&2
      return 4
    fi
    sleep 2
  done

  echo "[INFO] Downloading result: ${tag} -> ${out_mp4}"
  curl -sS -L "${BASE_URL}/v1/tasks/${task_id}/result" -o "${out_mp4}"
  echo "[INFO]   saved: ${out_mp4}"
}

echo "[INFO] BASE_URL=${BASE_URL}"
echo "[INFO] REF_VIDEO=${REF_VIDEO}"
echo "[INFO] REF_IMAGE=${REF_IMAGE}"
echo "[INFO] AUDIO_WAV=${AUDIO_WAV}"
echo "[INFO] OUT_DIR=${OUT_DIR}"

# Resolution-specific defaults (recommended)
SHIFT_480="${SHIFT_480:-7}"
SHIFT_720="${SHIFT_720:-11}"

T1_ID="$(create_task "V2V-480" "${REF_VIDEO}" "infinitetalk-480" "${SHIFT_480}" "ultimate_v2v_480")"
T2_ID="$(create_task "V2V-720" "${REF_VIDEO}" "infinitetalk-720" "${SHIFT_720}" "ultimate_v2v_720")"
T3_ID="$(create_task "I2V-480" "${REF_IMAGE}" "infinitetalk-480" "${SHIFT_480}" "ultimate_i2v_480")"
T4_ID="$(create_task "I2V-720" "${REF_IMAGE}" "infinitetalk-720" "${SHIFT_720}" "ultimate_i2v_720")"

poll_and_download "V2V-480" "${T1_ID}" "${OUT_DIR}/ultimate_v2v_480.mp4"
poll_and_download "V2V-720" "${T2_ID}" "${OUT_DIR}/ultimate_v2v_720.mp4"
poll_and_download "I2V-480" "${T3_ID}" "${OUT_DIR}/ultimate_i2v_480.mp4"
poll_and_download "I2V-720" "${T4_ID}" "${OUT_DIR}/ultimate_i2v_720.mp4"

echo "[INFO] All done. Results are in: ${OUT_DIR}"

