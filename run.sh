#!/usr/bin/env bash
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

# Use a local virtual environment in .venv, creating it if necessary
if [ ! -d ".venv" ]; then
    if command -v python3 >/dev/null 2>&1; then
        python3 -m venv .venv || {
            echo "Failed to create virtual environment in .venv" >&2
            exit 1
        }
    else
        python -m venv .venv || {
            echo "Failed to create virtual environment in .venv" >&2
            exit 1
        }
    fi
fi

# Activate the virtual environment
# shellcheck source=/dev/null
. ".venv/bin/activate"

# Run main.py inside the virtual environment
python -X utf8 main.py "$@"
