@echo off
chcp 65001 >nul
title 疲劳驾驶检测系统
echo ========================================
echo   疲劳驾驶检测系统 - 正在启动...
echo ========================================
echo.
cd /d "%~dp0"
python start.py
pause
