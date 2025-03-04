FROM linuxserver/webtop:debian-kde

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_PRIORITY=high

RUN apt update
RUN apt -y upgrade
RUN mkdir -p /usr/share/man/man1
RUN apt -y install openjdk-17-jre-headless 
RUN apt -y install python3-pip python3-tk python3-dev gnome-screenshot wget net-tools unzip
RUN apt -y install --no-install-recommends libreoffice

RUN python3 -m pip config set global.break-system-packages true
RUN python3 -m pip install pyautogui uvicorn fastapi python-multipart pillow pyscreeze numpy scipy pandas[excel] sympy bio matplotlib seaborn geopy
RUN apt -y install xdotool

COPY api.py /api.py
RUN mkdir -p /custom-cont-init.d && \
    echo $'#!/bin/bash\nHOME=/config DISPLAY=:1 python3 /api.py &\necho "API started"' > /custom-cont-init.d/start.sh && \
    chmod +x /custom-cont-init.d/start.sh
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /config /home/user