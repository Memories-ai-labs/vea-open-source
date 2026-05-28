#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PORT=8000
COMMAND=""
FRONTEND_DEV=false
SKIP_DASHBOARD_BUILD=false

PIDS_DIR="$ROOT_DIR/.dev"
FRONTEND_PID_FILE="$PIDS_DIR/frontend.pid"

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
DIM='\033[2m'
NC='\033[0m'

print_header() {
  echo
  echo -e "${CYAN}  ╔═══════════════════════════════════════╗${NC}"
  echo -e "${CYAN}  ║       VEA Developer Bootstrap         ║${NC}"
  echo -e "${CYAN}  ╚═══════════════════════════════════════╝${NC}"
  echo
}

info() {
  echo -e "${BLUE}$1${NC}"
}

success() {
  echo -e "${GREEN}  ✓ $1${NC}"
}

warn() {
  echo -e "${YELLOW}  ! $1${NC}"
}

fail() {
  echo -e "${RED}  ✗ $1${NC}"
}

error() {
  echo -e "${RED}$1${NC}" >&2
}

usage() {
  cat <<EOF
Usage:
  ./dev.sh [setup|up|doctor|down] [options]

Commands:
  setup               Install dependencies, prompt for API keys, build dashboard
  up                  Start backend (runs setup first if needed)
  doctor              Check all tooling and config without changing anything
  down                Stop background processes started by this script

Options:
  --port=PORT         Backend port (default: 8000)
  --frontend-dev      Also start Vite dev server on port 5173 (hot reload)
  --skip-dashboard-build
                      Skip dashboard production build
  --help, -h          Show this help

Behavior:
  - With no command: runs setup if uninitialized, otherwise up.
  - The dashboard is served by FastAPI at /app when dashboard/dist exists.
  - DaVinci Resolve Studio is optional but needed for preview rendering.
EOF
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

run_python() {
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    "$ROOT_DIR/.venv/bin/python" "$@"
  elif command_exists uv; then
    uv run python "$@"
  else
    python3 "$@"
  fi
}

require_command() {
  local cmd="$1"
  local help_text="$2"
  if ! command_exists "$cmd"; then
    fail "Missing required command: $cmd"
    echo "    $help_text" >&2
    exit 1
  fi
}

ensure_dev_dir() {
  mkdir -p "$PIDS_DIR"
}

read_config_value() {
  local key="$1"
  run_python - "$key" <<'PY'
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
  run_python - "$key" "$value" <<'PY'
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
    success "$key already configured"
    return
  fi

  if [[ ! -t 0 ]]; then
    if [[ "$required" == "true" ]]; then
      fail "Missing required config value for $key and no interactive TTY is available."
      exit 1
    fi
    warn "Skipping optional key $key (no interactive prompt available)"
    return
  fi

  echo
  if [[ "$required" == "true" ]]; then
    echo -e "  ${YELLOW}[required] $prompt_text${NC}"
  else
    echo -e "  ${DIM}[optional] $prompt_text${NC}"
  fi

  local value=""
  while true; do
    if [[ "$secret" == "true" ]]; then
      read -r -s -p "  > " value
      echo
    else
      read -r -p "  > " value
    fi

    if [[ -n "$value" ]]; then
      write_config_value "$key" "$value"
      success "Saved $key"
      break
    fi

    if [[ "$required" == "true" ]]; then
      warn "$key is required."
    else
      warn "Leaving $key unchanged."
      break
    fi
  done
}

copy_config_if_missing() {
  if [[ -f "$ROOT_DIR/config.json" ]]; then
    success "config.json present"
    return
  fi

  if [[ ! -f "$ROOT_DIR/config.example.json" ]]; then
    fail "config.example.json is missing"
    exit 1
  fi

  cp "$ROOT_DIR/config.example.json" "$ROOT_DIR/config.json"
  success "Created config.json from config.example.json"
}

repo_needs_setup() {
  [[ ! -d "$ROOT_DIR/.venv" || ! -f "$ROOT_DIR/config.json" || ! -d "$ROOT_DIR/dashboard/dist" ]]
}

# ── Checks ──────────────────────────────────────────────────────────────

check_python_version() {
  local version
  version="$(run_python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  local major minor
  major="$(echo "$version" | cut -d. -f1)"
  minor="$(echo "$version" | cut -d. -f2)"
  if [[ "$major" -ge 3 && "$minor" -ge 12 ]]; then
    success "Python $version"
  else
    fail "Python $version found, but 3.12+ is required"
    exit 1
  fi
}

check_core_tooling() {
  info "[check] Required tools"
  require_command uv "Install uv: https://docs.astral.sh/uv/getting-started/installation/"
  success "uv"
  check_python_version
  require_command ffmpeg "Install ffmpeg: brew install ffmpeg"
  success "ffmpeg"
  require_command node "Install Node.js 20+: https://nodejs.org"
  success "node $(node --version 2>/dev/null)"
  require_command npm "Install npm with Node.js."
  success "npm"
  require_command gcloud "Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install"
  success "gcloud"
}

check_gcloud_auth() {
  info "[check] Google Cloud credentials"
  if gcloud auth application-default print-access-token >/dev/null 2>&1; then
    success "Application-default credentials active"
  else
    warn "Google application-default credentials not found"
    echo -e "    Run: ${CYAN}gcloud auth application-default login${NC}"
    if [[ -t 0 ]]; then
      read -r -p "    Run it now? [Y/n] " response
      response="${response:-Y}"
      if [[ "$response" =~ ^[Yy]$ ]]; then
        gcloud auth application-default login
      fi
    fi
  fi
}

check_resolve() {
  info "[check] DaVinci Resolve (optional — needed for preview rendering)"

  # Check if Resolve process is running
  if pgrep -if "DaVinci Resolve" >/dev/null 2>&1 || pgrep -if "resolve" >/dev/null 2>&1; then
    success "DaVinci Resolve is running"
  else
    warn "DaVinci Resolve is not running"
    echo -e "    Preview rendering requires Resolve Studio running (GUI or -nogui mode)."
    echo -e "    Without it, FCPXML files will still be generated but not auto-rendered."
    return
  fi

  # Check scripting API paths
  local api_path="${RESOLVE_SCRIPT_API:-}"
  if [[ -z "$api_path" ]]; then
    # Try macOS default
    local mac_default="/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting"
    if [[ -d "$mac_default" ]]; then
      success "Resolve scripting API found at default macOS path"
    else
      warn "RESOLVE_SCRIPT_API not set and default path not found"
      echo "    Add to your shell profile:"
      echo '    export RESOLVE_SCRIPT_API="/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting"'
      echo '    export RESOLVE_SCRIPT_LIB="/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so"'
      echo '    export PYTHONPATH="$PYTHONPATH:$RESOLVE_SCRIPT_API/Modules/"'
    fi
  else
    if [[ -d "$api_path" ]]; then
      success "RESOLVE_SCRIPT_API set: $api_path"
    else
      warn "RESOLVE_SCRIPT_API set but directory not found: $api_path"
    fi
  fi

  # Try to connect via Python
  local resolve_status
  resolve_status="$(run_python - <<'PY' 2>/dev/null || echo "error"
import sys, os
api = os.environ.get("RESOLVE_SCRIPT_API", "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting")
modules = os.path.join(api, "Modules")
if modules not in sys.path:
    sys.path.insert(0, modules)
try:
    import DaVinciResolveScript as dvr
    r = dvr.scriptapp("Resolve")
    if r is None:
        print("no-connection")
    else:
        pm = r.GetProjectManager()
        if pm is None:
            print("free-edition")
        else:
            print("studio-ok")
except ImportError:
    print("no-import")
except Exception as e:
    print(f"error:{e}")
PY
)"

  case "$resolve_status" in
    studio-ok)
      success "Resolve Studio scripting API connected"
      ;;
    free-edition)
      warn "Resolve is running but scripting requires Studio edition (not Free)"
      ;;
    no-connection)
      warn "Resolve scripting returned None — try restarting Resolve"
      ;;
    no-import)
      warn "Could not import DaVinciResolveScript — check PYTHONPATH includes Modules/"
      ;;
    *)
      warn "Resolve check returned: $resolve_status"
      ;;
  esac
}

check_config_keys() {
  info "[check] API keys in config.json"
  if [[ ! -f "$ROOT_DIR/config.json" ]]; then
    fail "config.json missing"
    return
  fi

  local key val placeholder
  local -a keys=("OPENROUTER_API_KEY" "ELEVENLABS_API_KEY" "MEMORIES_API_KEY")
  local -a placeholders=("" "your-elevenlabs-api-key" "your-memories-ai-api-key")
  local -a labels=("OpenRouter (LLM + lvmm-core local video understanding)" "ElevenLabs (TTS + STT)" "Memories.ai (legacy V1 only)")
  local -a required=("true" "true" "false")

  for i in "${!keys[@]}"; do
    key="${keys[$i]}"
    val="$(read_config_value "$key")"
    placeholder="${placeholders[$i]}"
    if ! is_placeholder_value "$val" "$placeholder"; then
      success "${labels[$i]}: configured"
    else
      if [[ "${required[$i]}" == "true" ]]; then
        fail "${labels[$i]}: NOT SET"
      else
        warn "${labels[$i]}: not set (optional)"
      fi
    fi
  done
}

check_vinet_model() {
  info "[check] ViNet saliency model (optional — for dynamic cropping)"
  if [[ -f "$ROOT_DIR/vinet_v2/final_models/ViNet.pt" || -f "$ROOT_DIR/vinet_v2/final_models/vinet.pt" ]]; then
    success "ViNet model present"
    return
  fi

  warn "ViNet model weights not found"
  if [[ -t 0 ]]; then
    read -r -p "    Download/setup ViNet now? [y/N] " response
    if [[ "$response" =~ ^[Yy]$ ]]; then
      uv run python -m lib.utils.vinet_setup
      success "ViNet setup complete"
    else
      warn "Skipped ViNet setup (dynamic cropping will be unavailable)"
    fi
  fi
}

# ── Setup ───────────────────────────────────────────────────────────────

ensure_uv_env() {
  info "[setup] Syncing Python environment with uv..."
  uv sync
  success "Python dependencies installed"
}

ensure_dashboard_deps() {
  info "[setup] Installing dashboard dependencies..."
  if [[ -f "$ROOT_DIR/dashboard/package-lock.json" ]]; then
    (cd "$ROOT_DIR/dashboard" && npm ci --silent)
  else
    (cd "$ROOT_DIR/dashboard" && npm install --silent)
  fi
  success "Dashboard dependencies ready"
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
    info "[setup] Building dashboard..."
    (cd "$ROOT_DIR/dashboard" && npm run build 2>&1 | tail -1)
    success "Dashboard built → served at /app"
  else
    success "Dashboard build up to date"
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
  info "[setup] API keys"
  copy_config_if_missing

  prompt_for_config_key \
    "OPENROUTER_API_KEY" \
    "OpenRouter API key (https://openrouter.ai — LLM agent + lvmm-core + music generation):" \
    "" \
    "true" \
    "true"

  prompt_for_config_key \
    "ELEVENLABS_API_KEY" \
    "ElevenLabs API key (https://elevenlabs.io — narration TTS + STT):" \
    "your-elevenlabs-api-key" \
    "true" \
    "true"

  prompt_for_config_key \
    "MEMORIES_API_KEY" \
    "Memories.ai API key (optional — legacy V1 only):" \
    "your-memories-ai-api-key" \
    "false" \
    "true"

  prompt_for_config_key \
    "GOOGLE_CLOUD_PROJECT" \
    "Google Cloud Project ID (optional — only if using Vertex AI instead of OpenRouter):" \
    "your-gcp-project-id" \
    "false" \
    "false"

  prompt_for_config_key \
    "GOOGLE_CLOUD_LOCATION" \
    "Google Cloud location (press Enter for us-central1):" \
    "us-central1" \
    "false" \
    "false"
}

# ── Up / Down ───────────────────────────────────────────────────────────

check_port_available() {
  local port="$1"
  if lsof -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    fail "Port $port is already in use."
    echo "    Stop the process on that port or use --port=<other>" >&2
    exit 1
  fi
}

start_frontend_dev() {
  ensure_dev_dir
  if [[ -f "$FRONTEND_PID_FILE" ]]; then
    local existing_pid
    existing_pid="$(cat "$FRONTEND_PID_FILE")"
    if kill -0 "$existing_pid" >/dev/null 2>&1; then
      warn "Frontend dev server already running (pid $existing_pid)"
      return
    fi
    rm -f "$FRONTEND_PID_FILE"
  fi

  info "[up] Starting Vite dev server on http://localhost:5173 ..."
  (
    cd "$ROOT_DIR/dashboard"
    npm run dev -- --host 0.0.0.0
  ) >"$PIDS_DIR/frontend.log" 2>&1 &
  echo "$!" >"$FRONTEND_PID_FILE"
  success "Dashboard dev server started (http://localhost:5173)"
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

print_ready_message() {
  echo
  echo -e "${GREEN}  ┌─────────────────────────────────────┐${NC}"
  echo -e "${GREEN}  │         Environment is ready         │${NC}"
  echo -e "${GREEN}  └─────────────────────────────────────┘${NC}"
  echo
  echo -e "  Backend:   ${CYAN}http://localhost:${PORT}${NC}"
  echo -e "  API docs:  ${CYAN}http://localhost:${PORT}/docs${NC}"
  echo -e "  Dashboard: ${CYAN}http://localhost:${PORT}/app${NC}"
  if [[ "$FRONTEND_DEV" == "true" ]]; then
    echo -e "  Vite HMR:  ${CYAN}http://localhost:5173${NC}"
  fi
  echo
  echo -e "  ${DIM}Getting started:${NC}"
  echo -e "  ${DIM}  1. Open the dashboard${NC}"
  echo -e "  ${DIM}  2. Create a project${NC}"
  echo -e "  ${DIM}  3. Drop video files into data/workspaces/<project>/footage/${NC}"
  echo -e "  ${DIM}  4. Click into the project to index videos and start editing${NC}"
  echo
}

# ── Commands ────────────────────────────────────────────────────────────

run_setup() {
  print_header
  check_core_tooling
  echo
  ensure_repo_dirs
  ensure_uv_env
  echo
  prompt_for_config
  echo
  check_gcloud_auth
  echo
  check_resolve
  echo
  ensure_dashboard_deps
  build_dashboard_if_needed
  echo
  echo -e "${GREEN}  Setup complete. Run ${CYAN}./dev.sh up${GREEN} to start.${NC}"
  echo
}

run_doctor() {
  print_header
  check_core_tooling
  echo

  info "[check] Repository state"
  if [[ -d "$ROOT_DIR/.venv" ]]; then
    success ".venv exists"
  else
    fail ".venv missing — run ./dev.sh setup"
  fi

  if [[ -d "$ROOT_DIR/dashboard/node_modules" ]]; then
    success "dashboard/node_modules exists"
  else
    fail "dashboard/node_modules missing — run ./dev.sh setup"
  fi

  if [[ -d "$ROOT_DIR/dashboard/dist" ]]; then
    success "dashboard/dist exists"
  else
    warn "dashboard/dist missing (will be built on ./dev.sh up)"
  fi
  echo

  check_config_keys
  echo
  check_gcloud_auth
  echo
  check_resolve
  echo

  echo -e "  ${DIM}Doctor complete.${NC}"
  echo
}

run_down() {
  print_header
  stop_pid_file "$FRONTEND_PID_FILE" "frontend dev server"
  echo "  Background processes stopped."
}

run_up() {
  print_header
  check_core_tooling
  echo

  if repo_needs_setup; then
    warn "Repo not fully initialized — running setup first..."
    echo
    run_setup
    echo
  fi

  if [[ ! -f "$ROOT_DIR/config.json" ]]; then
    fail "config.json is missing."
    exit 1
  fi

  # Validate required keys
  local or_key
  or_key="$(read_config_value "OPENROUTER_API_KEY")"
  if [[ -z "$or_key" ]]; then
    local gcp_project
    gcp_project="$(read_config_value "GOOGLE_CLOUD_PROJECT")"
    if is_placeholder_value "$gcp_project" "your-gcp-project-id"; then
      fail "Neither OPENROUTER_API_KEY nor GOOGLE_CLOUD_PROJECT is configured — need at least one LLM provider"
      echo "    Run: ./dev.sh setup"
      exit 1
    fi
  fi

  local el_key
  el_key="$(read_config_value "ELEVENLABS_API_KEY")"
  if is_placeholder_value "$el_key" "your-elevenlabs-api-key"; then
    warn "ELEVENLABS_API_KEY not configured — narration TTS and clip refinement STT will be unavailable"
  fi

  build_dashboard_if_needed

  if [[ "$FRONTEND_DEV" == "true" ]]; then
    start_frontend_dev
  fi

  check_port_available "$PORT"
  print_ready_message
  echo -e "  ${GREEN}Starting backend. Press Ctrl+C to stop.${NC}"
  echo

  exec uv run python -m src.app
}

# ── Argument parsing ────────────────────────────────────────────────────

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
