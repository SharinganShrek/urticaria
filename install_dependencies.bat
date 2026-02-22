@echo off
REM Install all project dependencies. Double-click or run: install_dependencies.bat
cd /d "%~dp0"

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

echo Activating virtual environment and installing dependencies...
call .venv\Scripts\activate.bat
pip install -r requirements.txt

echo.
echo Done! You can now run:
echo   python topic_analysis_keywords.py
echo   python unmet_needs_analysis.py
pause
