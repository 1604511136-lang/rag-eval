#!/usr/bin/env bash
# One-shot setup on the cloud server (run as root, from /opt/rag-eval).
set -euo pipefail

APP_DIR=/opt/rag-eval

if [ "$(id -u)" -ne 0 ]; then
  echo "Please run as root: sudo bash deploy/setup.sh"
  exit 1
fi

echo "==> Check Python"
if ! command -v python3 >/dev/null; then
  echo "ERROR: python3 not found. Install with: apt install -y python3 python3-venv python3-pip"
  exit 1
fi
python3 --version

cd "$APP_DIR"

echo "==> Create venv and install dependencies"
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip -q
pip install -r requirements.txt -q

echo "==> Check .env"
if [ ! -f .env ]; then
  cp .env.example .env
  echo ""
  echo "!! Please edit $APP_DIR/.env and fill OPENAI_API_KEY,"
  echo "!! then re-run: bash deploy/setup.sh"
  exit 1
fi

echo "==> Check data files"
if [ ! -f data/chunks/chunks_512.jsonl ] || [ ! -f data/chunks/embeddings_512.npy ]; then
  echo "Local data not found, running full pipeline (download + embed + eval)..."
  python scripts/run_all.py
fi

echo "==> Install systemd service"
cp deploy/rag-qa.service /etc/systemd/system/rag-qa.service
systemctl daemon-reload
systemctl enable --now rag-qa

sleep 2
IP=$(hostname -I | awk '{print $1}')
echo ""
echo "==> Deploy done. Visit: http://${IP}:8000"
echo "    status: systemctl status rag-qa"
echo "    logs:   journalctl -u rag-qa -f"
