#!/bin/bash

if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: This script only works on macOS"
    exit 1
fi
if ! command -v brew &> /dev/null; then
    echo "Error: Homebrew is not installed. Please install it first."
    echo "Visit https://brew.sh for installation instructions."
    exit 1
fi
echo "System checks passed: macOS detected and Homebrew is installed."

brew install uv
if [ $? -ne 0 ]; then
    echo "Error: Failed to install UV package manager. Please check the error messages above."
    exit 1
fi

if [ ! -f "Makefile" ]; then
    echo "Error: Makefile not found in current directory, run this script from the root of the repository"
    exit 1
fi
if [ ! -d "tapeagents" ]; then
    echo "Error: tapeagents directory not found in current directory, run this script from the root of the repository"
    exit 1
fi
echo "Found required files and directories"

make setup
if [ $? -ne 0 ]; then
    echo "Error: Failed to install tapeagents. Please check the error messages above."
    exit 1
fi

brew install podman
echo "Podman installed successfully."
podman machine init
podman machine start
podman info
if [ $? -ne 0 ]; then
    echo "Error: Failed to initialize Podman. Please check the error messages above."
    exit 1
fi
echo "Podman initialized successfully."

if [ $? -ne 0 ]; then
    echo "Error: Failed to install tapeagents. Please check the error messages above."
    exit 1
fi

mkdir -p .pw-browsers
PLAYWRIGHT_BROWSERS_PATH=.pw-browsers uv run playwright install --with-deps chromium
if [ $? -ne 0 ]; then
    echo "Error: Failed to install Chromium. Please check the error messages above."
    exit 1
fi

brew install ffmpeg
if [ $? -ne 0 ]; then
    echo "Error: Failed to install ffmpeg. Please check the error messages above."
    exit 1
fi
ffmpeg -version
if [ $? -ne 0 ]; then
    echo "Error: Failed to verify ffmpeg installation. Please check the error messages above."
    exit 1
fi
echo "ffmpeg installed successfully."

echo "All dependencies installed successfully."
echo ""
echo "You can now run the agent with the following command: uv run -m examples.gaia_agent.scripts.browsergym_ui"