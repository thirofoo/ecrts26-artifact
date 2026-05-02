FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /artifact

ENV MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock uv.toml README.md ./
COPY src ./src
COPY experiments ./experiments
COPY external/RD-Gen ./external/RD-Gen
COPY reproduce.sh ./

RUN chmod +x ./reproduce.sh && uv sync --locked

CMD ["./reproduce.sh", "check"]
