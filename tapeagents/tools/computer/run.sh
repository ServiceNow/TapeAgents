#!/bin/bash

podman run \
    -v .:/home/computer/ \
    -p 5900:5900 \
    -p 8501:8501 \
    -p 6080:6080 \
    -p 8080:8080 \
    -it computer:latest