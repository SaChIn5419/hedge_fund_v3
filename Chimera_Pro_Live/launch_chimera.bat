@echo off
TITLE Chimera Launcher
set PYTHON="c:\Users\Sachin D B\anaconda3\python.exe"

echo ==========================================
echo    CHIMERA INSTITUTIONAL LAUNCHER
echo ==========================================
echo.
echo [1/2] Launching Neural Engine (Backend)...
start "Chimera Engine" cmd /k %PYTHON% main_ws.py

echo [2/2] Launching Cortex Dashboard (Frontend)...
start "Chimera Cortex" cmd /k %PYTHON% -m streamlit run tools/cortex_dashboard.py

echo.
echo 🚀 System is Active. 
echo - The Engine is running in a separate window.
echo - The Dashboard will open in your browser.
echo.
pause
