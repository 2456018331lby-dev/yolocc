@echo off
title YOLOCC - Garbage Detection System
cd /d "%~dp0"
echo.
echo  ========================================
echo   YOLO Garbage Classification System
echo  ========================================
echo.
echo   [1] Visual Dashboard (interactive)
echo   [2] Quick Demo (see results now)
echo   [3] Streamlit Web UI
echo   [4] API Server
echo   [5] Full Pipeline
echo   [6] Exit
echo.
set /p choice="  Choose [1-6]: "

if "%choice%"=="1" python launch.py
if "%choice%"=="2" python demo.py
if "%choice%"=="3" streamlit run deploy\app.py
if "%choice%"=="4" python -m src.api --weights weights\best.onnx
if "%choice%"=="5" python scripts\run_pipeline.py
if "%choice%"=="6" exit

pause
