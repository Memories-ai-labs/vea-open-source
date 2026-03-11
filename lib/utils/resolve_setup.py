"""DaVinci Resolve setup utilities — health check and environment configuration."""
from __future__ import annotations
import logging
import os
import platform
import shutil
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

MACOS_ENV = {
    "RESOLVE_SCRIPT_API": "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting",
    "RESOLVE_SCRIPT_LIB": "/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so",
}

LINUX_ENV = {
    "RESOLVE_SCRIPT_API": "/opt/resolve/Developer/Scripting",
    "RESOLVE_SCRIPT_LIB": "/opt/resolve/libs/Fusion/fusionscript.so",
}

MACOS_SETUP_INSTRUCTIONS = """
DaVinci Resolve Setup (macOS):

1. Add to your shell profile (~/.zshrc or ~/.bash_profile):

   export RESOLVE_SCRIPT_API="/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting"
   export RESOLVE_SCRIPT_LIB="/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so"
   export PYTHONPATH="$PYTHONPATH:$RESOLVE_SCRIPT_API/Modules/"

2. Start Resolve headless (add to launchd or run manually):

   /Applications/DaVinci\\ Resolve/DaVinci\\ Resolve.app/Contents/MacOS/resolve -nogui &

3. Requires DaVinci Resolve Studio (not the free version).
"""

LINUX_SETUP_INSTRUCTIONS = """
DaVinci Resolve Setup (Linux):

1. Install DaVinci Resolve Studio from blackmagicdesign.com

2. Start virtual display (required even in -nogui mode):

   Xvfb :99 -screen 0 1920x1080x24 &
   export DISPLAY=:99

3. Add to your shell profile:

   export RESOLVE_SCRIPT_API="/opt/resolve/Developer/Scripting"
   export RESOLVE_SCRIPT_LIB="/opt/resolve/libs/Fusion/fusionscript.so"
   export PYTHONPATH="$PYTHONPATH:$RESOLVE_SCRIPT_API/Modules/"

4. Start Resolve headless:

   DISPLAY=:99 /opt/resolve/bin/resolve -nogui &

5. Requires DaVinci Resolve Studio (not the free version).
"""


def check_resolve_status() -> dict:
    """
    Check whether DaVinci Resolve is available and running.

    Returns:
        dict with keys: running (bool), version (str|None), studio (bool), error (str|None)
    """
    # Check env vars
    api_path = os.environ.get("RESOLVE_SCRIPT_API")
    lib_path = os.environ.get("RESOLVE_SCRIPT_LIB")

    if not api_path or not lib_path:
        return {
            "running": False,
            "version": None,
            "studio": False,
            "error": "RESOLVE_SCRIPT_API and RESOLVE_SCRIPT_LIB environment variables not set. "
                     "Run lib/utils/resolve_setup.py for setup instructions.",
        }

    # Check PYTHONPATH contains Modules dir
    modules_path = os.path.join(api_path, "Modules")
    python_path = os.environ.get("PYTHONPATH", "")
    if modules_path not in python_path:
        import sys
        sys.path.insert(0, modules_path)

    # Try to import and connect
    try:
        import DaVinciResolveScript as dvr  # type: ignore
    except ImportError:
        return {
            "running": False,
            "version": None,
            "studio": False,
            "error": f"Could not import DaVinciResolveScript. Check RESOLVE_SCRIPT_API ({api_path}) "
                     "and ensure Resolve is running.",
        }

    try:
        resolve = dvr.scriptapp("Resolve")
    except Exception as e:
        return {
            "running": False,
            "version": None,
            "studio": False,
            "error": f"Could not connect to Resolve: {e}. Is Resolve running with -nogui?",
        }

    if resolve is None:
        return {
            "running": False,
            "version": None,
            "studio": False,
            "error": "Resolve returned None. Ensure Resolve is running (resolve -nogui) "
                     "and you have Resolve Studio (not Free).",
        }

    # Get version
    version = None
    try:
        fusion = resolve.Fusion()
        if fusion:
            version = str(fusion.GetAppInfo().get("Version", "unknown"))
    except Exception:
        pass

    # Check Studio by attempting project manager access (fails on Free)
    studio = False
    try:
        pm = resolve.GetProjectManager()
        studio = pm is not None
    except Exception:
        pass

    if not studio:
        return {
            "running": True,
            "version": version,
            "studio": False,
            "error": "Resolve is running but Studio features unavailable. "
                     "External scripting requires DaVinci Resolve Studio.",
        }

    return {
        "running": True,
        "version": version,
        "studio": True,
        "error": None,
    }


def print_setup_instructions() -> None:
    """Print platform-appropriate setup instructions."""
    if platform.system() == "Darwin":
        print(MACOS_SETUP_INSTRUCTIONS)
    else:
        print(LINUX_SETUP_INSTRUCTIONS)


def ensure_pythonpath() -> bool:
    """Add Resolve Modules to sys.path if not already there. Returns True if successful."""
    import sys

    api_path = os.environ.get("RESOLVE_SCRIPT_API")
    if not api_path:
        # Fall back to platform defaults
        if platform.system() == "Darwin":
            api_path = MACOS_ENV["RESOLVE_SCRIPT_API"]
            os.environ["RESOLVE_SCRIPT_API"] = api_path
            os.environ.setdefault("RESOLVE_SCRIPT_LIB", MACOS_ENV["RESOLVE_SCRIPT_LIB"])
        elif platform.system() == "Linux":
            api_path = LINUX_ENV["RESOLVE_SCRIPT_API"]
            os.environ["RESOLVE_SCRIPT_API"] = api_path
            os.environ.setdefault("RESOLVE_SCRIPT_LIB", LINUX_ENV["RESOLVE_SCRIPT_LIB"])
        else:
            return False

    modules_path = os.path.join(api_path, "Modules")
    if modules_path not in sys.path:
        sys.path.insert(0, modules_path)
    return os.path.isdir(modules_path)


if __name__ == "__main__":
    print("Checking DaVinci Resolve status...")
    status = check_resolve_status()
    if status["running"] and status["studio"]:
        print(f"Resolve is running (version: {status.get('version', 'unknown')})")
    else:
        print(f"Resolve not available: {status.get('error')}")
        print_setup_instructions()
