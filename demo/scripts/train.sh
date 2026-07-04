#!/usr/bin/env bash
# train.sh - 后台训练 + 自动关机
#
# Usage:
#   bash demo/scripts/train.sh debug    # 调试模式：报错不关机
#   bash demo/scripts/train.sh train    # 训练模式：结束/报错自动关机
#
set -euo pipefail
cd "$(dirname "$0")/../.."

MODE="${1:-debug}"   # 默认 debug
ROOT=demo
CONFIG=$ROOT/config/demo.yaml
OUT_DIR=/root/autodl-tmp/output/onereason_0.8b_sft
LOG=$OUT_DIR/train.log
VENV=$ROOT/LLaMA-Factory/.venv

# ==== 激活 venv ====
if [ -f "$VENV/Scripts/activate" ]; then
  source "$VENV/Scripts/activate"
elif [ -f "$VENV/bin/activate" ]; then
  source "$VENV/bin/activate"
fi

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=1

mkdir -p "$OUT_DIR"

echo "============================================"
echo " Mode: $MODE"
echo " Config: $CONFIG"
echo " Log: $LOG"
echo "============================================"

# ==== 启动训练（后台） ====
set +e  # 临时允许非零退出，方便捕获
llamafactory-cli train "$CONFIG" > "$LOG" 2>&1
EXIT_CODE=$?
set -e

echo ""
echo "============================================"
echo " Training exit code: $EXIT_CODE"
echo " Log saved to: $LOG"
echo "============================================"

# ==== 关机逻辑 ====
if [ "$MODE" = "train" ]; then
  if [ $EXIT_CODE -eq 0 ]; then
    echo "[INFO] Training finished successfully. Shutting down in 30s..."
    sleep 30
  else
    echo "[ERROR] Training failed with exit code $EXIT_CODE. Shutting down in 30s..."
    sleep 30
  fi

  # AutoDL 关机命令（按实际情况选一条）
  # 方式一：官方 API
  curl -s -X POST http://localhost:34201/api/v1/instance/stop 2>/dev/null || true
  # 方式二：系统关机
  shutdown -h now 2>/dev/null || poweroff 2>/dev/null || true
else
  if [ $EXIT_CODE -eq 0 ]; then
    echo "[INFO] Debug mode: training finished successfully (no shutdown)."
  else
    echo "[ERROR] Debug mode: training failed with exit code $EXIT_CODE (no shutdown)."
  fi
fi
