@echo off
chcp 65001 >nul
title 疲劳驾驶检测 Demo
echo ========================================
echo   疲劳驾驶检测 Demo - 正在启动...
echo ========================================
echo.
cd /d "%~dp0"
python demo_app.py
pause
