#!/bin/bash
if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: This script only works on macOS"
    exit 1
fi

# run podman and container
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
    podman machine set --user-mode-networking
    nohup podman machine start > /dev/null 2>&1
    echo "Podman machine started"
    podman info > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "Error: Failed to initialize Podman. Please check the error messages above."
        exit 1
    fi
fi
export DOCKER_HOST=http+unix://$(podman machine inspect --format '{{.ConnectionInfo.PodmanSocket.Path}}')
if ! podman images computer | grep -q "computer"; then
    echo "No computer image found, building one"
    podman images
    podman build -t computer:latest tapeagents/tools/computer/
    if [ $? -ne 0 ]; then
        echo "Failed to build computer image"
        exit 1
    fi
fi

# install dependencies
uv sync --all-extras

SCRIPT_DIR=$(dirname "$0")

# Add trap for cleanup
cleanup() {
    echo "Cleaning up..."
    pkill -f "$SCRIPT_DIR/chat.py"
    exit 0
}
trap cleanup SIGINT SIGTERM

mkdir -p .cache
# Initialize the log file
> /tmp/demo_stdout.log
uv run -m tapeagents.tools.computer
uv run -m http.server 8080 --directory $SCRIPT_DIR/static_content >> /tmp/demo_stdout.log 2>&1 &
REUSE_COMPUTER_CONTAINER=1 uv run -m streamlit run $SCRIPT_DIR/chat.py --server.headless true >> /tmp/demo_stdout.log 2>&1 &
sleep 3
echo "Tapeagents Operator is ready"
echo "Open http://localhost:8080 in your browser to begin"
if [[ "$OSTYPE" == "darwin"* ]]; then
    open http://localhost:8080
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open http://localhost:8080
fi
tail -n 100 -f /tmp/demo_stdout.log
