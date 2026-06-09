# Isolated development container for pilotless_poses
# Claude Code runs inside this container, safely separated from the host OS.
# The workspace is bind-mounted — edit files on the host with VS Code,
# changes are instantly visible inside the container.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DENO_INSTALL=/opt/deno
ENV NODE_HOME=/usr/local/node
ENV PATH="${DENO_INSTALL}/bin:${NODE_HOME}/bin:${PATH}"

# -------------------------------------------------------------------
# 1. System packages
# -------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    unzip \
    git \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3 -> python (many scripts expect 'python')
RUN ln -s /usr/bin/python3 /usr/local/bin/python

# -------------------------------------------------------------------
# 2. Node.js (needed by Claude Code)
# -------------------------------------------------------------------
ENV NODE_VERSION=22.12.0
RUN mkdir -p ${NODE_HOME} \
    && curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" \
    | tar -xJ -C ${NODE_HOME} --strip-components=1

RUN node --version && npm --version

# -------------------------------------------------------------------
# 3. Deno (project runtime)
# -------------------------------------------------------------------
RUN curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/opt/deno sh -s v2.7.1 \
    && chmod -R 755 /opt/deno

RUN deno --version

# -------------------------------------------------------------------
# 4. Claude Code CLI
# -------------------------------------------------------------------
RUN npm install -g @anthropic-ai/claude-code@2.1.169 \
    && claude --version

# -------------------------------------------------------------------
# 5. Python dependencies
# -------------------------------------------------------------------
COPY py-requirements.txt /tmp/py-requirements.txt
RUN pip3 install --break-system-packages -r /tmp/py-requirements.txt \
    && rm /tmp/py-requirements.txt

# -------------------------------------------------------------------
# 6. Create matching user (avoids permission issues with bind mount)
# -------------------------------------------------------------------
ARG HOST_UID=1000
ARG HOST_GID=1000
RUN userdel -r ubuntu 2>/dev/null || true \
    && groupadd -f -g ${HOST_GID} developer \
    && useradd -m -u ${HOST_UID} -g ${HOST_GID} -s /bin/bash developer

RUN apt-get update && apt-get install -y --no-install-recommends sudo \
    && echo "developer ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/developer \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER developer
WORKDIR /workspace

# -------------------------------------------------------------------
# 7. Entrypoint — keep the container alive indefinitely
# -------------------------------------------------------------------
ENTRYPOINT ["tail", "-f", "/dev/null"]
