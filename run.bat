@echo off
REM Windows helper to run main.py in the current Python environment (e.g. conda base with CUDA)

REM 自愈：如果缺少 TTS 依赖则自动安装（使用当前环境的 python）
python -c "import pyttsx3" >nul 2>nul
if errorlevel 1 (
    echo [INFO] Installing missing dependency: pyttsx3
    python -m pip install pyttsx3
)

REM 如果没有传参数，则默认使用 tts 配置；否则透传参数（如 --config voice）
if "%~1"=="" (
    REM 默认：使用 config.py 中的 tts 配置
    python main.py --config tts
) else (
    python main.py %*
)
