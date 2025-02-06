#!/bin/bash

echo "Starting Firefox on display :${DISPLAY_NUM}"
DISPLAY=:${DISPLAY_NUM} firefox-esr --no-remote --new-window about:blank -override ${HOME}/override.ini &
# Wait for Firefox process to start
sleep 2


