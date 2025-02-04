#!/bin/bash
set -e

./start_all.sh
./novnc_startup.sh

echo "Web VNC ready, http://localhost:6080/vnc.html?view_only=1&autoconnect=1&resize=scale"

python -m api.api > /tmp/api_stdout.log 2>&1 &
echo "Tool API ready, http://localhost:8000"

# STREAMLIT_SERVER_PORT=8501 python -m streamlit run tapeagents/examples/gaia_agent/scripts/streamlit.py &
echo "Streamlit ready, http://localhost:8501"

python http_server.py > /tmp/http_server_stdout.log &
echo "Tapeagents Operator is ready!"
echo "Open http://localhost:8080 in your browser to begin"
tail -f /tmp/api_stdout.log
