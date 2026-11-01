@echo off
title RAG QA Server
cd /d "C:\Users\x\rag-eval"

netstat -ano | findstr ":8000" | findstr "LISTENING" >nul
if %errorlevel%==0 (
    start "" http://127.0.0.1:8000
    exit /b
)

start "RAG QA Server" /min cmd /c "cd /d C:\Users\x\rag-eval && python -m uvicorn app:app --host 0.0.0.0 --port 8000"
timeout /t 3 /nobreak >nul
start "" http://127.0.0.1:8000
