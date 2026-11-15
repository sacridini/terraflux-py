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
# Build arguments:
#   PYTHON_VERSION  Python of the image (default 3.12)
#
# `docker build --target test -t terraflux-py:test .` builds an image that runs
# the test suite instead.

ARG PYTHON_VERSION=3.12

# ---- build: compile the C++ core into a wheel --------------------------------
FROM python:${PYTHON_VERSION}-slim AS build

RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY pyproject.toml CMakeLists.txt README.md LICENSE ./
COPY src ./src
# scikit-build-core fetches CMake, Ninja and pybind11; CMake fetches Eigen.
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .

# ---- base: the installed package -------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS base

# libgomp1: the OpenMP runtime of the compiled core; libexpat1: needed by the
# GDAL bundled in the rasterio wheel.
RUN apt-get update  && apt-get install -y --no-install-recommends libgomp1 libexpat1  && rm -rf /var/lib/apt/lists/*

COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir --find-links /wheels terraflux-py  && rm -rf /wheels

ENV PYTHONUNBUFFERED=1     MPLBACKEND=Agg

RUN useradd --create-home --uid 1000 terraflux

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

# the repository of the image on ghcr.io (its page, and the visibility it inherits)
LABEL org.opencontainers.image.source="https://github.com/sacridini/terraflux-py" \
      org.opencontainers.image.description="Terraflux: landscape ecology, connectivity and conservation planning" \
      org.opencontainers.image.licenses="GPL-3.0-or-later"

USER terraflux
WORKDIR /data
CMD ["python"]
