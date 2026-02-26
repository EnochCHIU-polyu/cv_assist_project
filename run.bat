@echo off
REM Windows helper to activate environment and run main.py
if not exist .venv (
    python -m venv .venv
)
call .venv\Scripts\activate
python main.py
