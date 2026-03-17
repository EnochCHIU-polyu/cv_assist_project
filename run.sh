#!/usr/bin/env bash
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

# Use current Python environment (e.g. conda with CUDA)
python -X utf8 main.py "$@"
