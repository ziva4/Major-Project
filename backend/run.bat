@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-inference\Scripts\python.exe" (
    py -3 -m venv --system-site-packages .venv-inference
    if errorlevel 1 exit /b 1
    .venv-inference\Scripts\python.exe -X utf8 -m pip install -r requirements-dashboard.txt
    if errorlevel 1 exit /b 1
)
echo Dashboard: http://127.0.0.1:5000
echo Press Ctrl+C to stop the camera and server.
.venv-inference\Scripts\python.exe -X utf8 app.py %*
pause
exit /b
