# Use the base image
FROM osrf/ros:humble-desktop-full-jammy

# Install necessary packages, including VNC server and full Ubuntu desktop environment
RUN apt-get update && apt-get install -y \
    ros-humble-turtlebot4-simulator \
    ubuntu-desktop \
    tightvncserver \
    dbus-x11 \
    x11-xserver-utils \
    && rm -rf /var/lib/apt/lists/*

# Set up VNC server with an initial configuration
RUN mkdir ~/.vnc && \
    echo "password" | vncpasswd -f > ~/.vnc/passwd && \
    chmod 600 ~/.vnc/passwd

# Create a startup script for the VNC server
RUN echo "#!/bin/bash\n\
export DISPLAY=:1\n\
vncserver :1 -geometry 1920x1080 -depth 24\n\
gnome-session &\n\
tail -f /dev/null" > /usr/local/bin/start_vnc.sh && \
    chmod +x /usr/local/bin/start_vnc.sh

# Expose VNC port (default port 5901)
EXPOSE 5901

# Start the VNC server and GNOME session
CMD ["/usr/local/bin/start_vnc.sh"]