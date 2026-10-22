@echo off
title Face Detection & Surveillance System
echo ========================================================
echo   Starting Surveillance Camera Wall & Face Recognition
echo ========================================================
echo.
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
