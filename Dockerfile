FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /tmp/build

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir . \
    && rm -rf /tmp/build

RUN useradd --create-home --shell /bin/sh openbench \
    && mkdir -p /work \
    && chown openbench:openbench /work

WORKDIR /work
USER openbench

ENTRYPOINT ["openbench"]
CMD ["--help"]
