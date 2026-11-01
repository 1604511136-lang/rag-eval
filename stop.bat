@echo off
set pid=
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do set pid=%%p
if defined pid (
    taskkill /f /pid %pid% >nul 2>&1
    echo RAG QA server stopped.
) else (
    echo RAG QA server is not running.
)
timeout /t 2 /nobreak >nul
