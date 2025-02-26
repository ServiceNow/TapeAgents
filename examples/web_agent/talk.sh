#!/bin/bash
uv run -m tapeagents.tools.computer
uv run -m http.server 8080 --directory $SCRIPT_DIR/static_content >> /tmp/talk_stdout.log 2>&1 &
uv run -m streamlit run examples/web_agent/talk_computer.py >> /tmp/talk_stdout.log 2>&1 &
echo "Open http://localhost:8080 in your browser to begin"
tail -f /tmp/talk_stdout.log
