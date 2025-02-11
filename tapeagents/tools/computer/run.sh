#!/bin/bash
rm -rf .computer_home
cp -r "$(dirname "$0")/home" .computer_home
podman run -p 3000:3000 -p 8000:8000 --mount type=bind,source=$(pwd)/.computer_home,target=/config -it computer /bin/bash -c "HOME=/config DISPLAY=:1 python3 /api.py"
