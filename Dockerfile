# --- Builder Image ---
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies and ffmpeg for trimming audio
RUN apt-get update && apt-get install -y curl \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && apt-get remove --purge -y curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


# Copy project files
COPY pyproject.toml poetry.lock* ./

# Create virtual environment and install dependencies
RUN uv venv .venv \
    && uv pip install --python .venv/bin/python .

# Copy the rest of the codebase
COPY . .

# --- Runtime Image ---
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install ffmpeg for pydub
RUN apt-get update && apt-get install -y ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy entire built app from builder stage
COPY --from=builder /app /app

# Ensure a fallback data dir exists inside container if volume not mounted
RUN mkdir -p /app/data

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Default entrypoint
CMD ["python", "-m", "src.pipelines.movieRecap.movieRecapPipeline"]



