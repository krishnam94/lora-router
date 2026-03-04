#!/bin/bash
# RunPod one-command setup for lora-router GPU benchmark.
#
# Prerequisites:
#   - RunPod RTX 4090 (24GB VRAM), PyTorch 2.4 template
#   - Container disk: 50GB, Volume disk: 20GB
#   - HuggingFace token with LLaMA-2 access (meta-llama/Llama-2-7b-hf)
#
# Usage:
#   # Set your HF token first
#   export HF_TOKEN="hf_..."
#
#   # Then run:
#   bash scripts/runpod_setup.sh
#
#   # Or one-liner from fresh pod:
#   curl -sSL https://raw.githubusercontent.com/krishnam94/lora-router/main/scripts/runpod_setup.sh | bash
#
# What this does:
#   1. Clones lora-router repo
#   2. Installs Python deps (torch, peft, transformers, etc.)
#   3. Downloads LLaMA-2-7B base model to persistent volume
#   4. Downloads 48 Styxxxx LoRA adapters
#   5. Downloads FLAN v2 test data from HuggingFace
#   6. Verifies GPU access and model loading
#
# Cost: ~$0.20/hr on RTX 4090 spot. Full setup takes ~15 min.

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() { echo -e "${GREEN}[INFO]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

# Paths - use RunPod persistent volume for models
WORKSPACE="/workspace"
REPO_DIR="${WORKSPACE}/lora-router"
MODEL_DIR="${WORKSPACE}/models/llama2-7b"
ADAPTER_DIR="${WORKSPACE}/adapters/flan_v2"
DATA_DIR="${REPO_DIR}/benchmarks/data"

echo "=============================================="
echo "  lora-router GPU Benchmark Setup"
echo "=============================================="
echo ""

# Step 0: Check HF token
if [ -z "${HF_TOKEN:-}" ]; then
    warn "HF_TOKEN not set. You need it for LLaMA-2 (gated model)."
    warn "Set it: export HF_TOKEN='hf_...'"
    warn "Get one: https://huggingface.co/settings/tokens"
    warn "Accept LLaMA-2 license: https://huggingface.co/meta-llama/Llama-2-7b-hf"
    echo ""
    read -p "Enter HF token (or press Enter to skip model download): " HF_TOKEN
    export HF_TOKEN
fi

# Step 1: Clone repo
info "Step 1/6: Clone repository"
if [ -d "${REPO_DIR}" ]; then
    info "  Repo exists, pulling latest..."
    cd "${REPO_DIR}" && git pull
else
    git clone https://github.com/krishnam94/lora-router.git "${REPO_DIR}"
    cd "${REPO_DIR}"
fi

# Step 2: Install dependencies
info "Step 2/6: Install Python dependencies"
pip install -e ".[eval,dev]" --quiet 2>&1 | tail -n 3
pip install accelerate --quiet

# Login to HuggingFace
if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential 2>/dev/null || true
fi

# Step 3: Download base model (to persistent volume)
info "Step 3/6: Download LLaMA-2-7B base model"
if [ -d "${MODEL_DIR}" ] && [ -f "${MODEL_DIR}/config.json" ]; then
    info "  Base model already downloaded, skipping"
else
    if [ -z "${HF_TOKEN:-}" ]; then
        warn "  Skipping base model download (no HF_TOKEN)"
    else
        mkdir -p "${MODEL_DIR}"
        python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'meta-llama/Llama-2-7b-hf',
    local_dir='${MODEL_DIR}',
    token='${HF_TOKEN}',
)
print('Base model downloaded successfully')
"
    fi
fi

# Step 4: Download 48 LoRA adapters
info "Step 4/6: Download 48 FLAN v2 LoRA adapters"
if [ -d "${ADAPTER_DIR}" ]; then
    # Check how many are already downloaded
    EXISTING=$(find "${ADAPTER_DIR}" -name "adapter_config.json" 2>/dev/null | wc -l | tr -d ' ')
    if [ "${EXISTING}" -ge 48 ]; then
        info "  All 48 adapters already downloaded, skipping"
    else
        info "  ${EXISTING}/48 adapters found, downloading remaining..."
        python scripts/download_flan_adapters.py --output-dir "${ADAPTER_DIR}"
    fi
else
    python scripts/download_flan_adapters.py --output-dir "${ADAPTER_DIR}"
fi

# Step 5: Download test data
info "Step 5/6: Download FLAN v2 test data"
if [ -f "${DATA_DIR}/combined_test.json" ]; then
    info "  Test data already exists, verifying..."
    python scripts/prepare_test_data.py --output "${DATA_DIR}/combined_test.json" --verify-only
else
    mkdir -p "${DATA_DIR}"
    python scripts/prepare_test_data.py --output "${DATA_DIR}/combined_test.json" --samples-per-task 50
fi

# Step 6: Verify GPU
info "Step 6/6: Verify GPU and environment"
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('  WARNING: No GPU detected!')

import transformers
print(f'  Transformers: {transformers.__version__}')

import peft
print(f'  PEFT: {peft.__version__}')
"

# Summary
echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "Paths:"
echo "  Repo:      ${REPO_DIR}"
echo "  Base model: ${MODEL_DIR}"
echo "  Adapters:  ${ADAPTER_DIR}"
echo "  Test data: ${DATA_DIR}/combined_test.json"
echo ""
echo "Quick start:"
echo "  cd ${REPO_DIR}"
echo ""
echo "  # Routing-only eval (fast, no model loading):"
echo "  python scripts/run_full_benchmark.py --routing-only \\"
echo "    --adapter-dir ${ADAPTER_DIR}"
echo ""
echo "  # Full eval with inference:"
echo "  python scripts/run_full_benchmark.py --all \\"
echo "    --base-model ${MODEL_DIR} \\"
echo "    --adapter-dir ${ADAPTER_DIR}"
echo ""
echo "  # Dry run (verify config without running):"
echo "  python scripts/run_full_benchmark.py --dry-run"
echo ""
