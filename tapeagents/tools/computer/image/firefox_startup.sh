#!/bin/bash

set -e

echo "Starting Firefox on display :${DISPLAY_NUM}"
firefox --display=:${DISPLAY_NUM} &

# Wait for Firefox process to start
sleep 2

# Check if Firefox is running
if pgrep -x "firefox" > /dev/null; then
    echo "Firefox started successfully"
    exit 0
else
    echo "Failed to start Firefox"
    exit 1
fi

