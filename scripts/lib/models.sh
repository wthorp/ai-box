# models.sh — model aliases for GGUF and EXL3 benchmark lanes.

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

model_tabby() {
  local quant=$1
  case "$quant" in
    GEMMA4_DENSE|GEMMA4_DENSE_EXL3|GEMINI_DENSE|GEMINI_DENSE_EXL3) echo "gemma-4-31b-dense-exl3" ;;
    GEMMA4_MOE_410|GEMMA4_MOE_410_EXL3|GEMINI_MOE_410|GEMINI_MOE_410_EXL3) echo "gemma-4-moe-exl3/gemma-4-moe-4.10bpw" ;;
    GEMMA4_MOE_510|GEMMA4_MOE_510_EXL3|GEMINI_MOE_510|GEMINI_MOE_510_EXL3) echo "gemma-4-moe-exl3/gemma-4-moe-5.10bpw" ;;
    GEMMA4_MOE|GEMMA4_MOE_EXL3|GEMINI_MOE|GEMINI_MOE_EXL3) echo "gemma-4-moe-exl3/gemma-4-moe-4.10bpw" ;;
    QWEN3_CODER_NEXT|QWEN3_CODER_NEXT_EXL3|QWEN_CODER_NEXT|QWEN_CODER_NEXT_EXL3) echo "qwen3-coder-next-4.0bpw" ;;
    *) echo "$quant" ;;
  esac
}

model_tool_format() {
  local quant=$1 model
  model=$(model_tabby "$quant")
  case "$model" in
    *[Gg]emma*|*[Gg]emini*) echo "gemma4" ;;
    *[Qq]wen*) echo "qwen3_coder" ;;
    *) echo "" ;;
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

model_tabby_exists() {
  local quant=$1
  local model d
  model=$(model_tabby "$quant") || return 1
  for d in "${MODELS_DIR:-/models}" /models; do
    [[ -d "${d}/${model}" ]] && return 0
  done
  return 1
}
