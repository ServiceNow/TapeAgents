#!/bin/bash
set -e

./start_all.sh
./novnc_startup.sh

python http_server.py > /tmp/server_logs.txt 2>&1 &

STREAMLIT_SERVER_PORT=8501 python -m streamlit run api/streamlit.py > /tmp/streamlit_stdout.log &

echo "Tapeagents Operator is ready!"
echo "Open http://localhost:8080 in your browser to begin"

tail -f /dev/null
