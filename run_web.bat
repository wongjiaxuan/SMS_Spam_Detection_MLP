@echo off
echo ============================================================
echo SMS Spam Detection System - Web UI
echo ============================================================
echo.

echo Installing web dependencies...
pip install flask

echo.
echo Starting Web Server...
echo ============================================================
echo.
echo 🌐 Open your browser at: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo ============================================================

python app.py
pause