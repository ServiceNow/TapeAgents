#!/bin/bash
pkill -f "python examples/web_agent/http_server.py"
pkill -f "examples/web_agent/talk_computer.py"
podman rm -f computer > /dev/null 2>&1
sleep 2
./tapeagents/tools/computer/run.sh >> /tmp/talk_stdout.log 2>&1 &
STREAMLIT_SERVER_PORT=8501 uv run -m streamlit run examples/web_agent/talk_computer.py >> /tmp/talk_stdout.log 2>&1 &
python examples/web_agent/http_server.py >> /tmp/talk_stdout.log 2>&1 &
echo "Open http://localhost:8080 in your browser to begin"
tail -f /tmp/talk_stdout.log
