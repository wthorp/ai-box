# models.sh — GGUF paths for UD quants (multi-part Q5/Q6/Q8 use part 1 in MODEL_FILE)

model_gguf() {
  local quant=$1
  case "$quant" in
    Q4|Q4_K_XL) echo "Qwen3-Coder-Next-UD-Q4_K_XL.gguf" ;;
    Q5|Q5_K_XL) echo "Qwen3-Coder-Next-UD-Q5_K_XL-00001-of-00003.gguf" ;;
    Q6|Q6_K_XL) echo "Qwen3-Coder-Next-UD-Q6_K_XL-00001-of-00003.gguf" ;;
    Q8|Q8_K_XL) echo "Qwen3-Coder-Next-UD-Q8_K_XL-00001-of-00003.gguf" ;;
    *) echo "Unknown quant: $quant" >&2; return 1 ;;
  esac
}

model_exists() {
  local quant=$1
  local f d
  f=$(model_gguf "$quant") || return 1
  for d in "${MODELS_DIR:-/models}" /models; do
    [[ -f "${d}/${f}" ]] && return 0
  done
  return 1
}
