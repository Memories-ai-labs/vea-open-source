#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PORT=8000
COMMAND=""
WITH_NGROK=false
FRONTEND_DEV=false
SKIP_DASHBOARD_BUILD=false

PIDS_DIR="$ROOT_DIR/.dev"
NGROK_PID_FILE="$PIDS_DIR/ngrok.pid"
FRONTEND_PID_FILE="$PIDS_DIR/frontend.pid"

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_header() {
  echo -e "${BLUE}================================================${NC}"
  echo -e "${BLUE}  VEA Developer Bootstrap${NC}"
  echo -e "${BLUE}================================================${NC}"
}

info() {
  echo -e "${BLUE}$1${NC}"
}

success() {
  echo -e "${GREEN}$1${NC}"
}

warn() {
  echo -e "${YELLOW}$1${NC}"
}

error() {
  echo -e "${RED}$1${NC}" >&2
}

usage() {
  cat <<EOF
Usage:
  ./dev.sh [setup|up|doctor|down] [options]

Commands:
  setup               Install repo dependencies, prompt for config, build dashboard
  up                  Start the backend and optional local helpers
  doctor              Check local tooling and repo readiness without changing anything
  down                Stop background processes started by this script

Options:
  --port=PORT         Backend port (default: 8000)
  --with-ngrok        Start ngrok and export MEMORIES_CAPTION_CALLBACK_URL
  --frontend-dev      Start Vite dev server on port 5173 in the background
  --skip-dashboard-build
                      Skip dashboard production build checks
  --help, -h          Show this help

Behavior:
  - With no command, the script runs setup if the repo is uninitialized, otherwise up.
  - The dashboard is served by FastAPI at /app when dashboard/dist exists.
  - ngrok is optional and only needed for the webhook-based v1 indexing flow.
EOF
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

require_command() {
  local cmd="$1"
  local help_text="$2"
  if ! command_exists "$cmd"; then
    error "Missing required command: $cmd"
    echo "  $help_text" >&2
    exit 1
  fi
}

ensure_dev_dir() {
  mkdir -p "$PIDS_DIR"
}

read_config_value() {
  local key="$1"
  python3 - "$key" <<'PY'
import json
import pathlib
import sys

key = sys.argv[1]
path = pathlib.Path("config.json")
if not path.exists():
    print("")
    raise SystemExit(0)

try:
    data = json.loads(path.read_text())
except Exception:
    print("")
    raise SystemExit(0)

value = data.get("api_keys", {}).get(key, "")
print("" if value is None else str(value))
PY
}

write_config_value() {
  local key="$1"
  local value="$2"
  python3 - "$key" "$value" <<'PY'
import json
import pathlib
import sys

key = sys.argv[1]
value = sys.argv[2]
path = pathlib.Path("config.json")
if not path.exists():
    raise SystemExit("config.json is missing")

data = json.loads(path.read_text())
api_keys = data.setdefault("api_keys", {})
api_keys[key] = value
path.write_text(json.dumps(data, indent=2) + "\n")
PY
}

is_placeholder_value() {
  local value="$1"
  local placeholder="$2"
  if [[ -z "$value" ]]; then
    return 0
  fi
  if [[ "$value" == "$placeholder" ]]; then
    return 0
  fi
  return 1
}

prompt_for_config_key() {
  local key="$1"
  local prompt_text="$2"
  local placeholder="$3"
  local required="$4"
  local secret="${5:-false}"

  local current_value
  current_value="$(read_config_value "$key")"

  if ! is_placeholder_value "$current_value" "$placeholder"; then
    success "  $key already configured"
    return
  fi

  if [[ ! -t 0 ]]; then
    if [[ "$required" == "true" ]]; then
      error "Missing required config value for $key and no interactive TTY is available."
      exit 1
    fi
    warn "  Skipping optional key $key (no interactive prompt available)"
    return
  fi

  echo
  if [[ "$required" == "true" ]]; then
    echo -e "${YELLOW}$prompt_text${NC}"
  else
    echo -e "${BLUE}$prompt_text${NC}"
  fi

  local value=""
  while true; do
    if [[ "$secret" == "true" ]]; then
      read -r -s -p "> " value
      echo
    else
      read -r -p "> " value
    fi

    if [[ -n "$value" ]]; then
      write_config_value "$key" "$value"
      success "  Saved $key to config.json"
      break
    fi

    if [[ "$required" == "true" ]]; then
      warn "  $key is required."
    else
      warn "  Leaving $key unchanged."
      break
    fi
  done
}

copy_config_if_missing() {
  if [[ -f "$ROOT_DIR/config.json" ]]; then
    success "  config.json present"
    return
  fi

  if [[ ! -f "$ROOT_DIR/config.example.json" ]]; then
    error "config.example.json is missing"
    exit 1
  fi

  cp "$ROOT_DIR/config.example.json" "$ROOT_DIR/config.json"
  success "  Created config.json from config.example.json"
}

repo_needs_setup() {
  [[ ! -d "$ROOT_DIR/.venv" || ! -f "$ROOT_DIR/config.json" || ! -d "$ROOT_DIR/dashboard/dist" ]]
}

check_core_tooling() {
  info "[doctor] Checking required developer tools..."
  require_command python3 "Install Python 3.12+ and ensure python3 is on PATH."
  require_command uv "Install uv: https://docs.astral.sh/uv/getting-started/installation/"
  require_command ffmpeg "Install ffmpeg (brew install ffmpeg, apt install ffmpeg, etc.)."
  require_command node "Install Node.js 20+."
  require_command npm "Install npm with Node.js."
  require_command gcloud "Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install"
  success "  Core tooling is available"
}

check_optional_tooling() {
  if [[ "$WITH_NGROK" == "true" ]]; then
    require_command ngrok "Install ngrok: https://ngrok.com/download"
    success "  ngrok is available"
  fi
}

ensure_uv_env() {
  info "[setup] Syncing Python environment with uv..."
  uv sync
  success "  Python dependencies installed"
}

ensure_dashboard_deps() {
  info "[setup] Installing dashboard dependencies..."
  if [[ -f "$ROOT_DIR/dashboard/package-lock.json" ]]; then
    (cd "$ROOT_DIR/dashboard" && npm ci)
  else
    (cd "$ROOT_DIR/dashboard" && npm install)
  fi
  success "  Dashboard dependencies ready"
}

dashboard_needs_build() {
  if [[ "$SKIP_DASHBOARD_BUILD" == "true" ]]; then
    return 1
  fi

  if [[ ! -d "$ROOT_DIR/dashboard/dist" ]]; then
    return 0
  fi

  if find "$ROOT_DIR/dashboard/src" -type f -newer "$ROOT_DIR/dashboard/dist" | grep -q .; then
    return 0
  fi

  if [[ -f "$ROOT_DIR/dashboard/package.json" && "$ROOT_DIR/dashboard/package.json" -nt "$ROOT_DIR/dashboard/dist" ]]; then
    return 0
  fi

  return 1
}

build_dashboard_if_needed() {
  if dashboard_needs_build; then
    info "[setup] Building dashboard bundle..."
    (cd "$ROOT_DIR/dashboard" && npm run build)
    success "  Dashboard built for FastAPI at /app"
  else
    success "  Dashboard build already up to date"
  fi
}

ensure_repo_dirs() {
  mkdir -p \
    "$ROOT_DIR/data/videos" \
    "$ROOT_DIR/data/indexing" \
    "$ROOT_DIR/data/outputs" \
    "$ROOT_DIR/data/workspaces" \
    "$ROOT_DIR/.cache"
}

prompt_for_config() {
  info "[setup] Checking config.json..."
  copy_config_if_missing

  prompt_for_config_key \
    "MEMORIES_API_KEY" \
    "Enter MEMORIES_API_KEY (required, from https://memories.ai/app/service/key):" \
    "your-memories-ai-api-key" \
    "true" \
    "true"

  prompt_for_config_key \
    "GOOGLE_CLOUD_PROJECT" \
    "Enter GOOGLE_CLOUD_PROJECT (required for Gemini / Vertex AI):" \
    "your-gcp-project-id" \
    "true" \
    "false"

  prompt_for_config_key \
    "GOOGLE_CLOUD_LOCATION" \
    "Enter GOOGLE_CLOUD_LOCATION or press Enter to keep the default us-central1:" \
    "us-central1" \
    "false" \
    "false"

  prompt_for_config_key \
    "ELEVENLABS_API_KEY" \
    "Enter ELEVENLABS_API_KEY or press Enter to skip narration setup:" \
    "your-elevenlabs-api-key" \
    "false" \
    "true"

  prompt_for_config_key \
    "SOUNDSTRIPE_KEY" \
    "Enter SOUNDSTRIPE_KEY or press Enter to skip music setup:" \
    "your-soundstripe-api-key" \
    "false" \
    "true"
}

check_gcloud_auth() {
  info "[doctor] Checking Google application-default credentials..."
  if gcloud auth application-default print-access-token >/dev/null 2>&1; then
    success "  Google application-default credentials are available"
  else
    warn "  Google application-default credentials are not ready."
    echo "  Run: gcloud auth application-default login"
  fi
}

check_vinet_model() {
  info "[setup] Checking ViNet model..."
  if [[ -f "$ROOT_DIR/vinet_v2/final_models/ViNet.pt" || -f "$ROOT_DIR/vinet_v2/final_models/vinet.pt" ]]; then
    success "  ViNet model already present"
    return
  fi

  warn "  ViNet model weights not found."
  if [[ -t 0 ]]; then
    read -r -p "  Download/setup ViNet now? [Y/n] " response
    response="${response:-Y}"
    if [[ "$response" =~ ^[Yy]$ ]]; then
      uv run python -m lib.utils.vinet_setup
      success "  ViNet setup complete"
    else
      warn "  Skipped ViNet setup"
    fi
  else
    warn "  Skipping ViNet setup (no interactive prompt available)"
  fi
}

check_port_available() {
  local port="$1"
  if lsof -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    error "Port $port is already in use."
    echo "  Stop the process on that port or run ./dev.sh up --port=<other-port>" >&2
    exit 1
  fi
}

ensure_ngrok_auth() {
  local has_auth=false
  local config_file=""
  for config_file in \
    "$HOME/.config/ngrok/ngrok.yml" \
    "$HOME/Library/Application Support/ngrok/ngrok.yml" \
    "$HOME/.ngrok2/ngrok.yml" \
    "$HOME/snap/ngrok/common/ngrok.yml"; do
    if [[ -f "$config_file" ]] && grep -q "authtoken:" "$config_file" 2>/dev/null; then
      has_auth=true
      break
    fi
  done

  if [[ "$has_auth" == "true" ]]; then
    success "  ngrok authenticated"
    return
  fi

  if [[ ! -t 0 ]]; then
    error "ngrok is not authenticated and no interactive TTY is available."
    exit 1
  fi

  warn "  ngrok needs an auth token."
  echo "  Get one from: https://dashboard.ngrok.com/get-started/your-authtoken"
  read -r -p "> " token
  if [[ -z "$token" ]]; then
    error "No ngrok auth token provided."
    exit 1
  fi
  ngrok config add-authtoken "$token" >/dev/null
  success "  ngrok authenticated"
}

start_ngrok() {
  ensure_dev_dir
  ensure_ngrok_auth

  local existing_url=""
  existing_url="$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | python3 - <<'PY'
import json
import sys

try:
    payload = json.loads(sys.stdin.read() or "{}")
except Exception:
    print("")
    raise SystemExit(0)

for tunnel in payload.get("tunnels", []):
    public_url = tunnel.get("public_url", "")
    if public_url.startswith("https://"):
        print(public_url)
        break
else:
    print("")
PY
)"

  if [[ -n "$existing_url" ]]; then
    export MEMORIES_CAPTION_CALLBACK_URL="${existing_url}/webhooks/memories/caption"
    success "  Reusing existing ngrok tunnel at $existing_url"
    return
  fi

  info "[up] Starting ngrok on port $PORT..."
  ngrok http "$PORT" >"$PIDS_DIR/ngrok.log" 2>&1 &
  local ngrok_pid=$!
  echo "$ngrok_pid" >"$NGROK_PID_FILE"

  local tunnel_url=""
  local retries=15
  while [[ $retries -gt 0 ]]; do
    sleep 1
    tunnel_url="$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | python3 - <<'PY'
import json
import sys

try:
    payload = json.loads(sys.stdin.read() or "{}")
except Exception:
    print("")
    raise SystemExit(0)

for tunnel in payload.get("tunnels", []):
    public_url = tunnel.get("public_url", "")
    if public_url.startswith("https://"):
        print(public_url)
        break
else:
    print("")
PY
)"
    if [[ -n "$tunnel_url" ]]; then
      break
    fi
    retries=$((retries - 1))
  done

  if [[ -z "$tunnel_url" ]]; then
    error "Could not determine ngrok public URL."
    exit 1
  fi

  export MEMORIES_CAPTION_CALLBACK_URL="${tunnel_url}/webhooks/memories/caption"
  success "  ngrok tunnel active at $tunnel_url"
}

start_frontend_dev() {
  ensure_dev_dir
  if [[ -f "$FRONTEND_PID_FILE" ]]; then
    local existing_pid
    existing_pid="$(cat "$FRONTEND_PID_FILE")"
    if kill -0 "$existing_pid" >/dev/null 2>&1; then
      warn "  Frontend dev server already running (pid $existing_pid)"
      return
    fi
    rm -f "$FRONTEND_PID_FILE"
  fi

  info "[up] Starting dashboard dev server on http://localhost:5173 ..."
  (
    cd "$ROOT_DIR/dashboard"
    npm run dev -- --host 0.0.0.0
  ) >"$PIDS_DIR/frontend.log" 2>&1 &
  echo "$!" >"$FRONTEND_PID_FILE"
  success "  Dashboard dev server started in background"
}

print_ready_message() {
  echo
  success "Environment is ready."
  echo "  Backend:   http://localhost:${PORT}"
  echo "  API docs:  http://localhost:${PORT}/docs"
  echo "  Dashboard: http://localhost:${PORT}/app"
  if [[ "$FRONTEND_DEV" == "true" ]]; then
    echo "  Frontend:  http://localhost:5173"
  fi
  if [[ "$WITH_NGROK" == "true" && -n "${MEMORIES_CAPTION_CALLBACK_URL:-}" ]]; then
    echo "  Callback:  ${MEMORIES_CAPTION_CALLBACK_URL}"
  fi
  echo
  echo "Next steps:"
  echo "  1. Open the dashboard URL above."
  echo "  2. Create or open a workspace."
  echo "  3. Drop footage into data/workspaces/<project>/footage/."
}

run_setup() {
  print_header
  check_core_tooling
  check_optional_tooling
  ensure_repo_dirs
  ensure_uv_env
  prompt_for_config
  check_gcloud_auth
  check_vinet_model
  ensure_dashboard_deps
  build_dashboard_if_needed
  echo
  success "Setup complete."
  echo "Run ./dev.sh up to start the backend."
}

run_doctor() {
  print_header
  check_core_tooling
  check_optional_tooling

  if [[ -d "$ROOT_DIR/.venv" ]]; then
    success "  .venv exists"
  else
    warn "  .venv is missing"
  fi

  if [[ -f "$ROOT_DIR/config.json" ]]; then
    success "  config.json exists"
  else
    warn "  config.json is missing"
  fi

  if [[ -d "$ROOT_DIR/dashboard/dist" ]]; then
    success "  dashboard/dist exists"
  else
    warn "  dashboard/dist is missing"
  fi

  check_gcloud_auth
  echo
  echo "Doctor completed."
}

stop_pid_file() {
  local file="$1"
  local label="$2"
  if [[ ! -f "$file" ]]; then
    return
  fi

  local pid
  pid="$(cat "$file")"
  if kill -0 "$pid" >/dev/null 2>&1; then
    kill "$pid" >/dev/null 2>&1 || true
    success "Stopped $label (pid $pid)"
  fi
  rm -f "$file"
}

run_down() {
  print_header
  stop_pid_file "$NGROK_PID_FILE" "ngrok"
  stop_pid_file "$FRONTEND_PID_FILE" "frontend dev server"
  echo "Background helper shutdown complete."
}

run_up() {
  print_header
  check_core_tooling
  check_optional_tooling

  if repo_needs_setup; then
    warn "Repo is not fully initialized. Running setup first..."
    run_setup
    echo
  fi

  if [[ ! -f "$ROOT_DIR/config.json" ]]; then
    error "config.json is missing."
    exit 1
  fi

  local memories_key
  memories_key="$(read_config_value "MEMORIES_API_KEY")"
  if is_placeholder_value "$memories_key" "your-memories-ai-api-key"; then
    error "MEMORIES_API_KEY is not configured in config.json"
    exit 1
  fi

  local gcp_project
  gcp_project="$(read_config_value "GOOGLE_CLOUD_PROJECT")"
  if is_placeholder_value "$gcp_project" "your-gcp-project-id"; then
    warn "GOOGLE_CLOUD_PROJECT is not configured. Gemini-backed features may fail."
  fi

  build_dashboard_if_needed

  if [[ "$WITH_NGROK" == "true" ]]; then
    start_ngrok
  else
    warn "Skipping ngrok. v1 webhook-based indexing will be unavailable."
  fi

  if [[ "$FRONTEND_DEV" == "true" ]]; then
    start_frontend_dev
  fi

  check_port_available "$PORT"
  print_ready_message
  echo
  success "Starting backend in foreground. Press Ctrl+C to stop."
  echo

  exec uv run python -m src.app
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    setup|up|doctor|down)
      COMMAND="$1"
      shift
      ;;
    --port=*)
      PORT="${1#*=}"
      shift
      ;;
    --with-ngrok)
      WITH_NGROK=true
      shift
      ;;
    --frontend-dev)
      FRONTEND_DEV=true
      shift
      ;;
    --skip-dashboard-build)
      SKIP_DASHBOARD_BUILD=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      error "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$COMMAND" ]]; then
  if repo_needs_setup; then
    COMMAND="setup"
  else
    COMMAND="up"
  fi
fi

case "$COMMAND" in
  setup) run_setup ;;
  up) run_up ;;
  doctor) run_doctor ;;
  down) run_down ;;
  *)
    error "Unknown command: $COMMAND"
    exit 1
    ;;
esac
