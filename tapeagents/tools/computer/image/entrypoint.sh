#!/bin/bash
set -e

./start_all.sh
./novnc_startup.sh


echo "Tapeagents Computer is ready!"
echo "Open http://localhost:6080/vnc.html?view_only=1&autoconnect=1&resize=scale in your browser to take a look"

python -m api.api | tee /tmp/api.log
tail -f /dev/null
