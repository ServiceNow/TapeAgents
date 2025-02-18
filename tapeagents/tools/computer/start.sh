#!/bin/bash
HOME=/config DISPLAY=:1 python3 /api.py > /tmp/api.log 2>&1 &
echo "API started"