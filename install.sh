#!/usr/bin/env bash

# Install project dependencies into the current Python environment (no virtualenv).
# Make sure you have activated your desired conda/pyenv environment before running this.
python -m pip install -r requirements.txt

echo "Dependencies installed into current environment."