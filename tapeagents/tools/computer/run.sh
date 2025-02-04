#!/bin/bash
podman rm -f computer
podman run \
    --name computer \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    -e SERPER_API_KEY=$SERPER_API_KEY \
    -p 5900:5900 \
    -p 6080:6080 \
    -p 8000:8000 \
    -it computer:latest