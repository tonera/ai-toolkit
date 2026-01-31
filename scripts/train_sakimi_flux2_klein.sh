#!/usr/bin/env bash
set -euo pipefail

# 一键训练（FLUX.2-klein LoRA）
# 运行前请确保：
# - 已在本项目 venv 中安装依赖（requirements.txt）
# - 具备 CUDA / Nvidia GPU 环境（ai-toolkit 主要面向此环境）
# - 若模型为 gated，需要在项目根目录放置 .env 并写入 HF_TOKEN=...

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${ROOT_DIR}/config/train_lora_flux2_klein_sakimi.yaml"

python3 "${ROOT_DIR}/scripts/prepare_sakimi_dataset.py" --dataset "/Users/zhangtao/training/sakimi"
python3 "${ROOT_DIR}/run.py" "${CONFIG_FILE}"

