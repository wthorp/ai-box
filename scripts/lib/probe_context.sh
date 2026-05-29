# probe_context.sh — Find largest context that loads without OOM.
# Sets PROBE_CTX and PROBE_VRAM on success.
# Usage: probe_max_context <service> <port> [moe] [min_ctx]

gpu_memory_csv() {
  local out
  out=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null \
        | head -1 | tr -d ' ')
  [[ "$out" =~ ^[0-9]+,[0-9]+$ ]] || return 1
  echo "$out"
}

# Wait until GPU memory.used is stable (avoids stale nvidia-smi after reload).
wait_vram_stable() {
  local min_used=${1:-8000}
  local min_elapsed=${2:-90}
  local max_wait=${3:-360}
  local elapsed=0 stable=0 last_used=0 used

  while [ $elapsed -lt "$max_wait" ]; do
    used=$(gpu_memory_csv | cut -d, -f1) || return 1
    if [ -n "$used" ] && [ "$used" -ge "$min_used" ]; then
      if [ "$used" -eq "$last_used" ]; then
        stable=$((stable + 1))
      else
        stable=0
      fi
      last_used=$used
      if [ $stable -ge 2 ] && [ $elapsed -ge "$min_elapsed" ]; then
        echo "$used"
        return 0
      fi
    else
      stable=0
      last_used=0
    fi
    sleep 10
    elapsed=$((elapsed + 10))
  done
  return 1
}

recreate_service() {
  local service=$1
  local models_dir="${COMPOSE_MODELS_DIR:-${MODELS_DIR:-/data/ai/models}}"
  MODELS_DIR="$models_dir" docker compose stop "$service" > /dev/null 2>&1 || true
  MODELS_DIR="$models_dir" docker compose rm -sf "$service" > /dev/null 2>&1 || true
  sleep 5
  MODELS_DIR="$models_dir" docker compose up -d --force-recreate "$service" > /dev/null 2>&1
}

probe_set_env_value() {
  local key=$1 value=$2 env_file=$3
  if declare -F set_env_value > /dev/null 2>&1; then
    set_env_value "$key" "$value" "$env_file"
  elif grep -q "^${key}=" "$env_file"; then
    sed -i "s|^${key}=.*|${key}=${value}|" "$env_file"
  else
    printf '%s=%s\n' "$key" "$value" >> "$env_file"
  fi
}

probe_max_context() {
  local service=$1 port=$2
  local moe=${3:-}
  local min_ctx=${4:-16384}
  local ai_box=${AI_BOX:-.}

  local -a all_candidates candidates
  if [[ "${ALLOW_SMALL_CTX:-0}" == "1" ]] || [[ "${MODEL_FILE:-}" =~ Q8 ]]; then
    all_candidates=(8192 16384 32768 65536 131072 262144)
  else
    all_candidates=(16384 32768 65536 131072 262144)
  fi
  # Always try largest first. The caller asked for max context that loads; a
  # VRAM threshold should reject underfilled successful loads, not change the
  # search into "first acceptable small context".
  for (( i=${#all_candidates[@]}-1; i>=0; i-- )); do
    ctx=${all_candidates[$i]}
    [[ "$ctx" -ge "$min_ctx" ]] && candidates+=("$ctx")
  done

  if [[ -n "${PROBE_MAX_CTX:-}" ]]; then
    local -a capped=()
    for ctx in "${candidates[@]}"; do
      [[ "$ctx" -le "${PROBE_MAX_CTX}" ]] && capped+=("$ctx")
    done
    candidates=("${capped[@]}")
  fi

  local ctx used total elapsed last_used probe_max=360
  [[ "${PROBE_MIN_VRAM_PCT:-0}" -gt 0 ]] && probe_max=300
  for ctx in "${candidates[@]}"; do
    if [[ -n "$moe" ]]; then
      probe_set_env_value CPU_MOE_LAYERS "$moe" "${ai_box}/.env"
      probe_set_env_value CONTEXT_SIZE "$ctx" "${ai_box}/.env"
    else
      probe_set_env_value CONTEXT_SIZE "$ctx" "${ai_box}/.env"
    fi
    recreate_service "$service"
    # Model load: Q6/Q8 need longer than Q4
    local load_wait=45
    [[ "${MODEL_FILE:-}" =~ Q6 ]] && load_wait=90
    [[ "${MODEL_FILE:-}" =~ Q8 ]] && load_wait=90
    sleep "$load_wait"
    elapsed=0
    stable=0
    last_used=0
    while [ $elapsed -lt "$probe_max" ]; do
      if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
        local mem
        mem=$(gpu_memory_csv) || return 1
        used=$(echo "$mem" | cut -d, -f1)
        total=$(echo "$mem" | cut -d, -f2)
        if [ "$used" -eq "$last_used" ] && [ "$used" -ge 8000 ]; then
          stable=$((stable + 1))
        else
          stable=0
        fi
        last_used=$used
        if [ $stable -ge 2 ] && [ $elapsed -ge 60 ]; then
          local min_pct=${PROBE_MIN_VRAM_PCT:-0}
          if [ "$min_pct" -gt 0 ] 2>/dev/null; then
            local pct_ok
            pct_ok=$(python3 -c "print(1 if 100*${used}/${total} >= ${min_pct} else 0)" 2>/dev/null || echo 0)
            if [ "$pct_ok" != "1" ]; then
              break
            fi
          fi
          PROBE_CTX=$ctx
          PROBE_VRAM="${used}/${total}"
          return 0
        fi
      else
        stable=0
        # No health after 2 min → this ctx won't load; try next candidate
        if [ $elapsed -ge 120 ]; then
          break
        fi
      fi
      sleep 10
      elapsed=$((elapsed + 10))
    done
  done
  return 1
}
