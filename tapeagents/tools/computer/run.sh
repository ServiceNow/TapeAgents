#!/bin/bash
rm -rf .computer
unzip -o $(dirname "$0")/home.zip -d .computer
podman run -p 3000:3000 -p 8000:8000 --mount type=bind,source=$(pwd)/.computer/home,target=/config -it computer /bin/bash -c "HOME=/config DISPLAY=:1 python3 /api.py"