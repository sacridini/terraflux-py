# syntax=docker/dockerfile:1
#
# Terraflux in a container: the package built from this source tree, with
# everything it uses.
#
#   docker build -t terraflux-py .
#   docker run --rm -it -v "$PWD:/data" terraflux-py               # Python shell
#   docker run --rm -v "$PWD:/data" terraflux-py python script.py  # a script in the current folder
#
# Each release publishes it as ghcr.io/sacridini/terraflux-py:<version> (amd64 and arm64), for
# Apptainer on a cluster:
#   apptainer pull terraflux.sif docker://ghcr.io/sacridini/terraflux-py:<version>
#   apptainer exec terraflux.sif python script.py
#
# Any user id can run it: on Linux, `--user "$(id -u):$(id -g)"` makes the files written in /data
# yours (Podman: `--userns=keep-id`). The cache (example data, downloads) is in the home folder,
# /home/terraflux/.cache: a named volume there keeps it between runs
# (`-v terraflux-cache:/home/terraflux/.cache`). No TERRAFLUX_CACHE is set in the image on purpose:
# Apptainer takes the image's variables, and a cache in the read-only .sif could not be written.
#
# Build arguments:
#   PYTHON_VERSION  Python of the image (default 3.12)
#   BUILD_JOBS      parallel compile jobs of the C++ core (default: the build tool's, every processor)
#   VERSION         the version in the image's labels (the release workflow passes pyproject's)
#
# `docker build --target test -t terraflux-py:test .` builds an image that runs
# the test suite instead.

ARG PYTHON_VERSION=3.12

# ---- build: compile the C++ core into a wheel --------------------------------
FROM python:${PYTHON_VERSION}-slim AS build

RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ \
 && rm -rf /var/lib/apt/lists/*

# Empty: the build tool's default (every processor); --build-arg BUILD_JOBS=4 caps the compiler.
ARG BUILD_JOBS=
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}

WORKDIR /src
COPY pyproject.toml CMakeLists.txt README.md LICENSE ./
COPY src ./src
# scikit-build-core fetches CMake, Ninja and pybind11; CMake fetches Eigen.
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .

# ---- base: the installed package -------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS base

# libgomp1: the OpenMP runtime of the compiled core; libexpat1: needed by the
# GDAL bundled in the rasterio wheel; tini: an init that passes signals on (below).
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libexpat1 tini \
 && rm -rf /var/lib/apt/lists/*

COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir --find-links /wheels terraflux-py && rm -rf /wheels

# HOME is set, and open to every user (sticky, as /tmp), so that a run with any user id
# (`docker run --user`, Podman) has a home, a cache and a configuration it can write.
# Apptainer replaces HOME with the user's own.
ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    HOME=/home/terraflux

RUN useradd --create-home --uid 1000 terraflux \
 && mkdir -p /home/terraflux/.cache /home/terraflux/.config \
 && chmod 1777 /home/terraflux /home/terraflux/.cache /home/terraflux/.config

# ---- test: the test suite against the installed package ----------------------
FROM base AS test

RUN pip install --no-cache-dir "pytest>=7" threadpoolctl
# The tests use the reference scripts in tools/ and the bundled maps by their
# path in the source tree.
COPY --chown=terraflux pyproject.toml /tests/pyproject.toml
COPY --chown=terraflux src/terraflux/data /tests/src/terraflux/data
COPY --chown=terraflux src/terraflux/bio/data /tests/src/terraflux/bio/data
COPY --chown=terraflux src/terraflux/cost/data /tests/src/terraflux/cost/data
COPY --chown=terraflux src/terraflux/surface/data /tests/src/terraflux/surface/data
COPY --chown=terraflux tools /tests/tools
COPY --chown=terraflux tests /tests/tests
USER terraflux
WORKDIR /tests
CMD ["pytest", "-q", "-p", "no:cacheprovider"]

# ---- runtime (default target) ------------------------------------------------
FROM base AS runtime

ARG VERSION=dev

# source: the repository of the image on ghcr.io (its page, and the visibility it inherits)
LABEL org.opencontainers.image.title="terraflux-py" \
      org.opencontainers.image.description="Terraflux's Python library: landscape ecology, connectivity and conservation planning" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.url="https://sacridini.github.io/terraflux/python/" \
      org.opencontainers.image.documentation="https://sacridini.github.io/terraflux/python/" \
      org.opencontainers.image.source="https://github.com/sacridini/terraflux-py" \
      org.opencontainers.image.licenses="GPL-3.0-or-later"

USER terraflux
WORKDIR /data
# tini as process 1 passes `docker stop` and Ctrl+C on to Python, which as process 1 would ignore
# SIGTERM (docker stop then waits 10 s and kills it); -s also keeps it quiet where it is not
# process 1 (`apptainer run`). `apptainer exec` does not use the entrypoint.
ENTRYPOINT ["/usr/bin/tini", "-s", "--"]
CMD ["python"]
