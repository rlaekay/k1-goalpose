#!/bin/bash
# Fetch training artifacts (checkpoints/logs/videos) from the training server.
#
# Code is no longer synced this way -- push locally with git and `git pull`
# on the server (see README.md). This script only exists because checkpoints/
# logs/videos are large binaries that don't belong in git (see .gitignore).

set -euo pipefail

SERVER="${SERVER:-<SERVER>}"
# Path to the git-cloned repo root on the server (contains htwk-gym/).
SERVER_REPO="${SERVER_REPO:-<SERVER_REPO>}"
LOCAL_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/htwk-gym"

echo "📥 Pulling training artifacts from $SERVER:$SERVER_REPO/htwk-gym/ ..."
mkdir -p "$LOCAL_PATH/checkpoints" "$LOCAL_PATH/logs" "$LOCAL_PATH/videos"
for d in checkpoints logs videos; do
    rsync -avz "$SERVER:$SERVER_REPO/htwk-gym/$d/" "$LOCAL_PATH/$d/" 2>/dev/null || true
done
echo "✅ Pull complete"
