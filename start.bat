@echo off
title Face Detection & Recognition System
echo ========================================================
echo   Starting Face Recognition & Motion Detection System
echo ========================================================
echo.
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
pause
