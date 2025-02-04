#!/bin/bash
podman rm -f computer
podman run \
    --name computer \
    -p 5900:5900 \
    -p 8501:8501 \
    -p 6080:6080 \
    -p 8080:8080 \
    -p 8000:8000 \
    -it computer:latest