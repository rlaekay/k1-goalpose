#!/bin/bash

SERVER="user@user-ESC4000A-E12"
SERVER_PATH="/mnt/DATA/workspace/ws_eungkyu/htwk-gym"
LOCAL_PATH="$(pwd)/htwk-gym"

if [ "$1" == "push" ]; then
    echo "📤 Pushing code to server..."
    rsync -avz --delete \
        --exclude="__pycache__" \
        --exclude=".git" \
        --exclude="checkpoints" \
        --exclude="logs" \
        --exclude="videos" \
        "$LOCAL_PATH/" "$SERVER:$SERVER_PATH/"
    echo "✅ Push complete"

elif [ "$1" == "pull" ]; then
    echo "📥 Pulling results from server..."
    mkdir -p "$LOCAL_PATH/checkpoints" "$LOCAL_PATH/logs" "$LOCAL_PATH/videos"
    rsync -avz \
        "$SERVER:$SERVER_PATH/checkpoints/" "$LOCAL_PATH/checkpoints/" 2>/dev/null || true
    rsync -avz \
        "$SERVER:$SERVER_PATH/logs/" "$LOCAL_PATH/logs/" 2>/dev/null || true
    rsync -avz \
        "$SERVER:$SERVER_PATH/videos/" "$LOCAL_PATH/videos/" 2>/dev/null || true
    echo "✅ Pull complete"

else
    echo "Usage: $0 [push|pull]"
    echo "  push: upload code to server"
    echo "  pull: download results from server"
fi
