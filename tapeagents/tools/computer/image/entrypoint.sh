#!/bin/bash
set -e

./start_all.sh
./novnc_startup.sh

echo "Web VNC ready, http://localhost:6080/vnc.html?view_only=1&autoconnect=1&resize=scale"

touch /tmp/api_stdout.log
python api.py > /tmp/api_stdout.log 2>&1 &
echo "Tool API ready, http://localhost:8000"
tail -f /tmp/api_stdout.log