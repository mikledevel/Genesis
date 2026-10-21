# Genesis AI agent-browser sandbox image - see app/core/agent_browser.py's module docstring
# for the full security model and, importantly, how this differs from (and is WEAKER than)
# the file-bridge isolation used by docker_sandbox.py's DockerSandboxNetworked for skills.
#
# UNLIKE sandbox.Dockerfile (network_mode="none", no real network ever), this container runs
# with real network access, because a browser has to actually fetch real web pages - there is
# no practical way to proxy a full browser session (HTML, sub-resources, JS, CSS, redirects,
# cookies) through a simple JSON request/response bridge the way a single API call can be.
# Isolation here instead comes from: an ephemeral, single-use container (destroyed after each
# session, never reused), no persistent volume beyond the session's own screenshots, a non-root
# user, and the resource/timeout/step-count limits enforced on the host side in
# app/core/agent_browser.py. This is a real, conscious security trade-off, not an oversight -
# see that file for what a stronger (MITM-proxy-based domain-filtering) version would need.
FROM python:3.12-slim

# System deps Chromium needs that aren't in the slim base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 \
    libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2 libpango-1.0-0 \
    libcairo2 fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir playwright==1.47.0 \
    && playwright install chromium \
    && playwright install-deps chromium

# Non-root user - same reasoning as sandbox.Dockerfile: whatever the agent does inside the
# browser, it never runs as root inside the container.
RUN useradd --create-home --uid 1000 agentuser
USER agentuser
ENV HOME=/home/agentuser

COPY agent_browser_harness.py /home/agentuser/harness.py

WORKDIR /session
ENTRYPOINT ["python", "/home/agentuser/harness.py"]
