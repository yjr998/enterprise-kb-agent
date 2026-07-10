@echo off
cd /d "%~dp0"

echo ========================================
echo Enterprise KB Agent
echo ========================================

if not exist "chroma_db" (
    echo [1/3] Building vector DB...
    python scripts\build_vectordb.py
    if errorlevel 1 (
        echo build_vectordb failed
        pause
        exit /b 1
    )
) else (
    echo [1/3] chroma_db OK
)

echo [2/3] Checking Ollama model qwen2.5:3b...
ollama list | findstr "qwen2.5:3b" >nul
if errorlevel 1 (
    echo Pulling qwen2.5:3b...
    ollama pull qwen2.5:3b
    if errorlevel 1 (
        echo ollama pull failed
        pause
        exit /b 1
    )
)

echo [3/3] Starting http://127.0.0.1:8000
python run.py
pause
