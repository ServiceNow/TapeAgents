#!/bin/bash
if ! podman ps --format "{{.Names}}" | grep -q "^computer$"; then
    echo "Starting new computer container..."
    ./tapeagents/tools/computer/run.sh > /tmp/demo_stdout.log 2>&1 &
else
    echo "Container 'computer' found"
    if [[ "$*" == *"--restart"* ]]; then
        echo "Restarting computer container..."
        ./tapeagents/tools/computer/run.sh > /tmp/demo_stdout.log 2>&1 &
    fi
fi
podman kill tapeagents-code-exec
pkill -f "tapeagents/tools/computer/image/http_server.py"
pkill -f "examples/gaia_agent/scripts/chat.py"
pkill -f "examples/gaia_agent/scripts/run_code_sandbox.py"
echo "Starting Code Sandbox..."
uv run examples/gaia_agent/scripts/run_code_sandbox.py &
echo "Starting Chat UI..."
python tapeagents/tools/computer/image/http_server.py > /tmp/demo_stdout.log 2>&1 &
STREAMLIT_SERVER_PORT=8501 uv run -m streamlit run examples/gaia_agent/scripts/chat.py --server.headless true > /tmp/demo_stdout.log 2>&1 &
sleep 2
echo "Tapeagents Operator is ready"
echo "Open http://localhost:8080 in your browser to begin"
if [[ "$OSTYPE" == "darwin"* ]]; then
    open http://localhost:8080
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open http://localhost:8080
fi
tail -f /tmp/demo_stdout.log
