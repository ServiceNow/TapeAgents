#!/bin/bash
pkill -f "python tapeagents/tools/computer/image/http_server.py"
pkill -f "streamlit run examples/gaia_agent/scripts/talk_computer.py"
STREAMLIT_SERVER_PORT=8501 uv run -m  streamlit run examples/gaia_agent/scripts/talk_computer.py &
python tapeagents/tools/computer/image/http_server.py > /tmp/http_server_stdout.log &
echo "Open http://localhost:8080 in your browser to begin"
wait
