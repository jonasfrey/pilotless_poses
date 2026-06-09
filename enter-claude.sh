#!/usr/bin/env bash
# =============================================================================
# enter-claude.sh — Drop into the isolated container and run Claude Code
#
# This script exec's into the running Docker container and launches
# Claude Code with --dangerously-skip-permissions inside the isolated
# environment. The container cannot access files outside the workspace
# or harm your host OS.
#
# Usage:
#   ./enter-claude.sh                  # Interactive Claude Code session
#   ./enter-claude.sh -p "your prompt" # One-shot prompt mode
#   ./enter-claude.sh --help           # Show Claude Code help
# =============================================================================

set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-pilotless-poses-isolated}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- helpers ---------------------------------------------------------------
red()   { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
yellow(){ printf '\033[33m%s\033[0m\n' "$*"; }

# ---- pre-flight checks -----------------------------------------------------
if ! command -v docker &>/dev/null; then
    red "Error: docker is not installed or not in PATH."
    exit 1
fi

if ! docker inspect "$CONTAINER_NAME" --format '{{.State.Running}}' &>/dev/null; then
    yellow "Container '$CONTAINER_NAME' is not running."

    # Check if the container exists but is stopped
    if docker inspect "$CONTAINER_NAME" --format '{{.State.Status}}' &>/dev/null; then
        yellow "Container exists but is stopped. Starting it..."
        docker start "$CONTAINER_NAME" >/dev/null
        # Give it a moment
        sleep 1
    else
        yellow "Container does not exist. Building and starting..."
        cd "$SCRIPT_DIR"
        docker compose up -d --build
        sleep 2
    fi
fi

# ---- launch Claude Code inside the container --------------------------------
green "Entering isolated container '$CONTAINER_NAME'..."
green "Workspace: /workspace"
green "Claude Code is sandboxed — only /workspace is shared with the host."
echo ""

# Source DeepSeek API credentials, then launch Claude Code.
# Pass all arguments through to claude.
exec docker exec -it "$CONTAINER_NAME" \
    bash -c 'source /workspace/deepseek_claude.txt && exec claude --dangerously-skip-permissions "$@"' -- "$@"
