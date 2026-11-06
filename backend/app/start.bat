
@echo off
title RoboResearch AI Launcher

echo Starting RoboResearch AI...

start "RoboResearch Backend" cmd /k "cd /d ""%~dp0backend"" && call .venv\Scripts\activate.bat && uvicorn app.main:app --reload"

start "RoboResearch Frontend" cmd /k "cd /d ""%~dp0frontend"" && npm run dev"

echo.
echo RoboResearch AI is launching!
echo Frontend: http://localhost:5173
echo Backend:  http://127.0.0.1:8000
echo.

pause
