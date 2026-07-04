ARG TABBYAPI_VERSION=latest
FROM ghcr.io/theroyallab/tabbyapi:${TABBYAPI_VERSION}

USER root
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential python3-dev && \
    rm -rf /var/lib/apt/lists/*
