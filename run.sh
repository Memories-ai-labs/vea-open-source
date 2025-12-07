#!/bin/bash
#
# VEA Playground - Run Script
#
# This script handles everything needed to run the VEA service:
# 1. Checks/installs dependencies if needed
# 2. Sets up ngrok for webhook callbacks
# 3. Starts the FastAPI server
#
# Usage:
#   ./run.sh              # Run the server
#   ./run.sh --setup-only # Just run setup, don't start server
#

set -e

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PORT=8000
SETUP_ONLY=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --setup-only)
            SETUP_ONLY=true
            ;;
        --port=*)
            PORT="${arg#*=}"
            ;;
        --help|-h)
            echo "Usage: ./run.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --setup-only   Only run setup, don't start the server"
            echo "  --port=PORT    Use a different port (default: 8000)"
            echo "  --help, -h     Show this help message"
            exit 0
            ;;
    esac
done

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  VEA Playground${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# =============================================================================
# STEP 1: Check/Setup Environment
# =============================================================================

check_setup() {
    echo -e "${BLUE}[1/4] Checking environment...${NC}"

    local needs_setup=false

    # Check virtual environment
    if [ ! -d ".venv" ]; then
        echo -e "  ${YELLOW}! Virtual environment not found${NC}"
        needs_setup=true
    else
        echo -e "  ${GREEN}✓ Virtual environment exists${NC}"
    fi

    # Check config.json
    if [ ! -f "config.json" ]; then
        echo -e "  ${YELLOW}! config.json not found${NC}"
        needs_setup=true
    else
        echo -e "  ${GREEN}✓ config.json exists${NC}"
    fi

    # Check ffmpeg
    if ! command -v ffmpeg &> /dev/null; then
        echo -e "  ${RED}✗ ffmpeg not installed${NC}"
        echo -e "    Install with: ${YELLOW}sudo apt install ffmpeg${NC}"
        exit 1
    else
        echo -e "  ${GREEN}✓ ffmpeg installed${NC}"
    fi

    if [ "$needs_setup" = true ]; then
        echo ""
        echo -e "${YELLOW}Setup required. Running setup...${NC}"
        run_setup
    fi
}

run_setup() {
    echo ""
    echo -e "${BLUE}Running first-time setup...${NC}"

    # Create virtual environment
    if [ ! -d ".venv" ]; then
        echo -e "  Creating virtual environment..."
        python3 -m venv .venv
    fi

    # Activate and install dependencies
    source .venv/bin/activate

    echo -e "  Installing dependencies..."
    pip install --quiet --upgrade pip

    if [ -f "requirements.txt" ]; then
        pip install --quiet -r requirements.txt
    fi

    # Install optional dependencies
    pip install --quiet "scenedetect[opencv]" psutil 2>/dev/null || true

    # Setup ViNet
    echo -e "  Setting up ViNet model..."
    python -m lib.utils.vinet_setup --force 2>/dev/null || echo -e "  ${YELLOW}(ViNet setup skipped)${NC}"

    # Create directories
    mkdir -p data/videos data/indexing data/outputs .cache

    # Create config.json from example
    if [ ! -f "config.json" ]; then
        if [ -f "config.example.json" ]; then
            cp config.example.json config.json
            echo ""
            echo -e "${YELLOW}================================================${NC}"
            echo -e "${YELLOW}  IMPORTANT: Edit config.json with your API keys!${NC}"
            echo -e "${YELLOW}================================================${NC}"
            echo ""
            echo -e "Required keys:"
            echo -e "  - MEMORIES_API_KEY (from https://memories.ai)"
            echo -e "  - GOOGLE_CLOUD_PROJECT (your GCP project ID)"
            echo ""
            echo -e "Then run this script again."
            exit 0
        fi
    fi

    echo -e "${GREEN}✓ Setup complete${NC}"
}

# =============================================================================
# STEP 2: Setup ngrok
# =============================================================================

setup_ngrok() {
    echo ""
    echo -e "${BLUE}[2/4] Setting up ngrok tunnel...${NC}"

    # Check if ngrok is installed
    if ! command -v ngrok &> /dev/null; then
        echo -e "  ${YELLOW}ngrok not found. Installing...${NC}"

        if command -v snap &> /dev/null; then
            sudo snap install ngrok
        else
            curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
            echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
            sudo apt update && sudo apt install -y ngrok
        fi

        if ! command -v ngrok &> /dev/null; then
            echo -e "  ${RED}✗ Failed to install ngrok${NC}"
            echo -e "  Install manually from: https://ngrok.com/download"
            exit 1
        fi
    fi
    echo -e "  ${GREEN}✓ ngrok installed${NC}"

    # Check authentication
    local has_auth=false
    for config_file in ~/.config/ngrok/ngrok.yml ~/snap/ngrok/common/ngrok.yml ~/.ngrok2/ngrok.yml; do
        if [ -f "$config_file" ] && grep -q "authtoken:" "$config_file" 2>/dev/null; then
            has_auth=true
            break
        fi
    done

    if [ "$has_auth" = false ]; then
        echo ""
        echo -e "  ${YELLOW}ngrok needs authentication${NC}"
        echo -e "  1. Sign up at: https://dashboard.ngrok.com/signup"
        echo -e "  2. Get token from: https://dashboard.ngrok.com/get-started/your-authtoken"
        echo ""
        read -p "  Paste your ngrok auth token: " NGROK_TOKEN

        if [ -n "$NGROK_TOKEN" ]; then
            ngrok config add-authtoken "$NGROK_TOKEN"
            echo -e "  ${GREEN}✓ ngrok authenticated${NC}"
        else
            echo -e "  ${RED}✗ No token provided${NC}"
            exit 1
        fi
    else
        echo -e "  ${GREEN}✓ ngrok authenticated${NC}"
    fi

    # Check if ngrok is already running with a tunnel we can reuse
    NGROK_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | grep -o '"public_url":"https://[^"]*' | head -1 | cut -d'"' -f4)

    if [ -n "$NGROK_URL" ]; then
        echo -e "  ${GREEN}✓ Reusing existing ngrok tunnel${NC}"
        export MEMORIES_CAPTION_CALLBACK_URL="${NGROK_URL}/webhooks/memories/caption"
        echo -e "  Public URL: ${BLUE}${NGROK_URL}${NC}"
        echo -e "  Callback URL: ${BLUE}${MEMORIES_CAPTION_CALLBACK_URL}${NC}"
        return
    fi

    # No existing tunnel, start a new one
    # Kill any stale ngrok processes first
    pkill -f "ngrok http" 2>/dev/null || true
    sleep 1

    # Start ngrok in background
    echo -e "  Starting ngrok tunnel on port ${PORT}..."
    ngrok http $PORT --log=stdout > /tmp/ngrok_vea.log 2>&1 &
    NGROK_PID=$!

    # Wait for ngrok to start
    sleep 3

    if ! kill -0 $NGROK_PID 2>/dev/null; then
        echo -e "  ${RED}✗ ngrok failed to start${NC}"
        cat /tmp/ngrok_vea.log
        exit 1
    fi

    # Get public URL
    local retries=5
    while [ $retries -gt 0 ]; do
        NGROK_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | grep -o '"public_url":"https://[^"]*' | head -1 | cut -d'"' -f4)
        if [ -n "$NGROK_URL" ]; then
            break
        fi
        sleep 1
        retries=$((retries - 1))
    done

    if [ -z "$NGROK_URL" ]; then
        echo -e "  ${RED}✗ Could not get ngrok URL${NC}"
        kill $NGROK_PID 2>/dev/null
        exit 1
    fi

    export MEMORIES_CAPTION_CALLBACK_URL="${NGROK_URL}/webhooks/memories/caption"
    echo -e "  ${GREEN}✓ ngrok tunnel active${NC}"
    echo -e "  Public URL: ${BLUE}${NGROK_URL}${NC}"
    echo -e "  Callback URL: ${BLUE}${MEMORIES_CAPTION_CALLBACK_URL}${NC}"

    # Store PID for cleanup
    echo $NGROK_PID > /tmp/ngrok_vea.pid
}

# =============================================================================
# STEP 3: Verify config
# =============================================================================

verify_config() {
    echo ""
    echo -e "${BLUE}[3/4] Verifying configuration...${NC}"

    source .venv/bin/activate

    # Quick Python check to verify API keys are loaded
    python3 -c "
import json
import sys

with open('config.json') as f:
    config = json.load(f)

api_keys = config.get('api_keys', {})
memories_key = api_keys.get('MEMORIES_API_KEY', '')

if not memories_key or memories_key == 'your-memories-ai-api-key':
    print('  \033[0;31m✗ MEMORIES_API_KEY not configured\033[0m')
    print('    Edit config.json and add your Memories.ai API key')
    sys.exit(1)
else:
    print('  \033[0;32m✓ MEMORIES_API_KEY configured\033[0m')

gcp_project = api_keys.get('GOOGLE_CLOUD_PROJECT', '')
if not gcp_project or gcp_project == 'your-gcp-project-id':
    print('  \033[1;33m! GOOGLE_CLOUD_PROJECT not set (Gemini will be unavailable)\033[0m')
else:
    print('  \033[0;32m✓ GOOGLE_CLOUD_PROJECT configured\033[0m')
"

    if [ $? -ne 0 ]; then
        exit 1
    fi
}

# =============================================================================
# STEP 4: Start server
# =============================================================================

start_server() {
    echo ""
    echo -e "${BLUE}[4/4] Starting VEA server...${NC}"
    echo ""
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}  Server starting on http://localhost:${PORT}${NC}"
    echo -e "${GREEN}================================================${NC}"
    echo -e "  ngrok inspector: http://localhost:4040"
    echo -e "  API docs: http://localhost:${PORT}/docs"
    echo ""
    echo -e "  Press Ctrl+C to stop"
    echo ""

    source .venv/bin/activate
    python -m src.app
}

# =============================================================================
# Cleanup handler
# =============================================================================

cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${NC}"

    # Kill ngrok if we started it
    if [ -f /tmp/ngrok_vea.pid ]; then
        kill $(cat /tmp/ngrok_vea.pid) 2>/dev/null || true
        rm /tmp/ngrok_vea.pid
    fi

    exit 0
}

trap cleanup SIGINT SIGTERM

# =============================================================================
# Main
# =============================================================================

check_setup

if [ "$SETUP_ONLY" = true ]; then
    echo ""
    echo -e "${GREEN}Setup complete. Run ./run.sh to start the server.${NC}"
    exit 0
fi

setup_ngrok
verify_config
start_server
