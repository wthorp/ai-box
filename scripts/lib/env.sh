#!/bin/bash
# Helpers for mutating .env during tuning runs without losing the user's config.

save_env_file() {
  local env_file=${1:-.env}
  local snapshot
  snapshot=$(mktemp)
  cp "$env_file" "$snapshot"
  echo "$snapshot"
}

restore_env_file() {
  local snapshot=$1 env_file=${2:-.env}
  [[ -f "$snapshot" ]] || return 0
  cp "$snapshot" "$env_file"
  rm -f "$snapshot"
}

set_env_value() {
  local key=$1 value=$2 env_file=${3:-.env}
  if grep -q "^${key}=" "$env_file"; then
    sed -i "s|^${key}=.*|${key}=${value}|" "$env_file"
  else
    printf '%s=%s\n' "$key" "$value" >> "$env_file"
  fi
}

env_value() {
  local key=$1 env_file=${2:-.env}
  grep "^${key}=" "$env_file" | tail -1 | cut -d= -f2-
}
