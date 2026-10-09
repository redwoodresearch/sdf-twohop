#!/usr/bin/env bash
# Publish the chat dashboard privately on the Redwood tailnet: Streamlit listens on loopback only and
# Tailscale Serve (through the devbox's redwood-app helper) terminates HTTPS. Ctrl-C stops both.
#   scripts/chat_dashboard.sh [local-port=8501] [https-port=443]
set -euo pipefail
cd "$(dirname "$0")/.."
port="${1:-8501}"
https_port="${2:-443}"
exec redwood-app run "$port" "$https_port" -- sh -c \
  'exec uv run streamlit run scripts/chat_dashboard.py --server.port "$1" \
     --browser.serverAddress "$REDWOOD_APP_HOSTNAME" --browser.serverPort "$2"' \
  _ "$port" "$https_port"
