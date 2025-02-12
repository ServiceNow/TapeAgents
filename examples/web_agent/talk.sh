#!/bin/bash
pkill -f "python tapeagents/tools/computer/image/http_server.py"
pkill -f "examples/gaia_agent/scripts/talk_computer.py"
sleep 2
STREAMLIT_SERVER_PORT=8501 uv run -m streamlit run examples/gaia_agent/scripts/talk_computer.py > /tmp/talk_stdout.log 2>&1 &
python tapeagents/tools/computer/image/http_server.py > /tmp/talk_stdout.log 2>&1 &
echo "Open http://localhost:8080 in your browser to begin"
tail -f /tmp/talk_stdout.log
