#!/bin/bash
podman kill tapeagents-code-exec
pkill -f "python tapeagents/tools/computer/image/http_server.py"
pkill -f "streamlit run examples/gaia_agent/scripts/streamlit.py"
echo "Starting Code Sandbox..."
uv run examples/gaia_agent/scripts/run_code_sandbox.py &
echo "Starting Chat UI..."
STREAMLIT_SERVER_PORT=8501 uv run -m  streamlit run examples/gaia_agent/scripts/streamlit.py &
python tapeagents/tools/computer/image/http_server.py > /tmp/http_server_stdout.log &
echo "Tapeagents Operator is ready!"
echo "Open http://localhost:8080 in your browser to begin"
wait
