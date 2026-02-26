@echo off
REM Windows helper to activate environment and run main.py
if not exist .venv (
    python -m venv .venv
    .venv\Scripts\python -m pip install --upgrade pip
    .venv\Scripts\python -m pip install -r requirements.txt
)
call .venv\Scripts\activate

REM 自愈：如果缺少 TTS 依赖则自动安装
python -c "import pyttsx3" >nul 2>nul
if errorlevel 1 (
    echo [INFO] Installing missing dependency: pyttsx3
    python -m pip install pyttsx3
)

if "%~1"=="" (
    REM 默认：使用 config.py 中的 tts 配置
    python main.py --config tts
) else (
    REM 允许透传自定义参数，例如: run.bat --config voice
    python main.py %*
)
