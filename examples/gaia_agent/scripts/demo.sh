#!/bin/bash

# Add trap for cleanup without stopping podman machine
cleanup() {
    echo "Cleaning up..."
    podman kill tapeagents-code-exec 2>/dev/null
    pkill -f "tapeagents/tools/computer/image/http_server.py"
    pkill -f "examples/gaia_agent/scripts/chat.py"
    pkill -f "examples/gaia_agent/scripts/run_code_sandbox.py"
    exit 0
}

# Set up trap for SIGINT and SIGTERM
trap cleanup SIGINT SIGTERM

if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: This script only works on macOS"
    exit 1
fi

if ! command -v podman &> /dev/null; then
    echo "Podman is not installed, installing..."

    if ! command -v brew &> /dev/null; then
        echo "Error: Homebrew is not installed. Please install it first."
        echo "Visit https://brew.sh for installation instructions."
        exit 1
    fi
    brew install podman
    echo "Podman installed"
    podman machine init > /dev/null 2>&1
    echo "Podman initialized"
fi
if ! podman machine list | grep -q "Currently running"; then
    nohup podman machine start > /dev/null 2>&1
    echo "Podman machine started"
    podman info > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "Error: Failed to initialize Podman. Please check the error messages above."
        exit 1
    fi
fi
if ! podman ps --format "{{.Names}}" | grep -q "^computer$"; then
    echo "No computer container found, starting one"
    if ! podman images computer | grep -q "computer"; then
        echo "No computer image found, building one"
        podman images
        podman build -t computer:latest tapeagents/tools/computer/
        if [ $? -ne 0 ]; then
            echo "Failed to build computer image"
            exit 1
        fi
    else
        echo "Computer image found"
    fi
    echo "Starting computer container"
    nohup podman run --rm --name computer -p 5900:5900 -p 6080:6080 -p 8000:8000 -it computer:latest > /tmp/computer.log 2>&1 &
    echo -n "Waiting for computer container to start"
    # Wait up to 15 seconds for computer container to be running
    for i in {1..15}; do
        if podman ps | grep -q "computer"; then
            break
        fi
        sleep 1
        echo -n "."
        if [ $i -eq 15 ]; then
            echo "Error: Computer container failed to start within 15 seconds"
            exit 1
        fi
    done
    echo "."
    echo -n "Waiting for API init"
    while ! grep -q "Uvicorn running on http://0.0.0.0:8000" /tmp/computer.log; do
        sleep 1
        echo -n "."
    done
    echo "."
    echo "Computer started"
fi

echo "Starting Code Sandbox..."
uv run examples/gaia_agent/scripts/run_code_sandbox.py &
echo "Starting Chat UI..."
export GROUNDING_API_URL="https://snow-llmd-grounding-8000.job.console.elementai.com"

uv run tapeagents/tools/computer/image/http_server.py > /tmp/demo_stdout.log 2>&1 &
STREAMLIT_SERVER_PORT=8501 uv run -m streamlit run examples/gaia_agent/scripts/chat.py --server.headless true > /tmp/demo_stdout.log 2>&1 &
sleep 2
echo "Tapeagents Operator is ready"
echo "Open http://localhost:8080 in your browser to begin"
if [[ "$OSTYPE" == "darwin"* ]]; then
    open http://localhost:8080
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open http://localhost:8080
fi
tail -n 100 -f /tmp/demo_stdout.log
