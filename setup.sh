#!/usr/bin/env bash
# setup.sh — Bootstrap a vanilla Ubuntu 26.04 server for AI workloads.
# Installs Docker (pinned), NVIDIA driver utils, CUDA toolkit, and the
# NVIDIA Container Toolkit, then tunes the OS for inference performance.
#
# Usage: sudo bash setup.sh

set -euo pipefail

# ── preflight ────────────────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
  echo "ERROR: run as root or with sudo." >&2
  exit 1
fi

. /etc/os-release
if [[ "$ID" != "ubuntu" || "$VERSION_ID" != "26.04" ]]; then
  echo "WARNING: tested on Ubuntu 26.04 — detected $PRETTY_NAME. Proceeding anyway."
fi

echo "==> [1/7] System update"
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get upgrade -y -q
apt-get install -y -q --no-install-recommends \
  curl gnupg ca-certificates software-properties-common

# ── Docker ───────────────────────────────────────────────────────────────────
echo "==> [2/7] Docker"
if ! command -v docker &>/dev/null; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg

  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    > /etc/apt/sources.list.d/docker.list

  apt-get update -q
  apt-get install -y -q \
    docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin

  # Pin Docker so apt upgrade never changes it unexpectedly
  apt-mark hold docker-ce docker-ce-cli containerd.io \
                docker-buildx-plugin docker-compose-plugin

  systemctl enable --now docker
else
  echo "    docker already installed, skipping."
fi

# Add invoking user to docker group if called via sudo
if [[ -n "${SUDO_USER:-}" ]]; then
  usermod -aG docker "$SUDO_USER"
  echo "    added $SUDO_USER to docker group (re-login to take effect)."
fi

# ── NVIDIA driver utils ───────────────────────────────────────────────────────
echo "==> [3/7] NVIDIA driver utils (nvidia-utils-595-server)"
apt-get install -y -q nvidia-utils-595-server

# ── CUDA toolkit ─────────────────────────────────────────────────────────────
echo "==> [4/7] CUDA toolkit 13.1"
if [[ ! -f /etc/apt/sources.list.d/cuda.list ]]; then
  curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub \
    | gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg
  echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] \
https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" \
    > /etc/apt/sources.list.d/cuda.list
  apt-get update -q
fi
apt-get install -y -q cuda-toolkit-13-1

# ── NVIDIA Container Toolkit ─────────────────────────────────────────────────
echo "==> [5/7] NVIDIA Container Toolkit"
if ! command -v nvidia-ctk &>/dev/null; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -q
  apt-get install -y -q nvidia-container-toolkit
else
  echo "    nvidia-ctk already installed, skipping."
fi

echo "==> [6/7] Configure Docker NVIDIA runtime and CDI"
nvidia-ctk runtime configure --runtime=docker
nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
systemctl restart docker

# ── OS-level performance tuning ───────────────────────────────────────────────
echo "==> [7/7] OS tuning"
cat > /etc/sysctl.d/99-ai-box.conf <<'EOF'
# Reduce swap pressure — keep model weights in RAM
vm.swappiness = 10
# Larger dirty page writeback window — less I/O jitter during inference
vm.dirty_ratio = 20
vm.dirty_background_ratio = 5
EOF
sysctl -p /etc/sysctl.d/99-ai-box.conf

# ── done ─────────────────────────────────────────────────────────────────────
echo ""
echo "==> Setup complete. Verification:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
docker info 2>/dev/null | grep -iE "runtime|nvidia|cdi" || true
echo ""
echo "NOTE: If your user was added to the docker group, run: newgrp docker"
echo "      or log out and back in before using docker without sudo."
