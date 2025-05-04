FROM python:3.12-slim AS builder

WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y curl build-essential \
 && curl -LsSf https://astral.sh/uv/install.sh | sh \
 && mv /root/.local/bin/uv /usr/local/bin/uv \
 && apt-get remove --purge -y curl build-essential \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml poetry.lock* ./

# Create venv and install deps using system uv
RUN uv venv .venv \
 && uv pip install --python .venv/bin/python .

# Copy rest of the codebase
COPY . .

# --- Runtime Image ---
FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=builder /app /app

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

CMD ["python", "-m", "src.app"]
