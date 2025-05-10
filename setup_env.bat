@echo off
cd /d %~dp0
echo 📦 Creating fresh virtual environment...
rmdir /s /q venv
"%LOCALAPPDATA%\Programs\Python\Python311\python.exe" -m venv venv

if not exist venv\Scripts\activate.bat (
    echo ❌ Failed to create virtual environment.
    pause
    exit /b
)

echo ✅ Environment created.

echo 🧪 Activating...
call venv\Scripts\activate.bat

echo 📥 Installing required packages...
pip install --upgrade pip
pip install tensorflow==2.13.0 keras==2.13.1 retina-face flask opencv-python pillow matplotlib pandas scikit-learn

echo ✅ Done installing.

echo 🛠 Creating .vscode/launch.json for VS Code...
mkdir .vscode >nul 2>&1

(
echo {
echo   "version": "0.2.0",
echo   "configurations": [
echo     {
echo       "name": "Python: Training",
echo       "type": "python",
echo       "request": "launch",
echo       "program": "${workspaceFolder}/Training.py",
echo       "console": "integratedTerminal",
echo       "python": "${workspaceFolder}/venv/Scripts/python.exe"
echo     }
echo   ]
echo }
) > .vscode\launch.json

echo ✅ VS Code configured to use venv!
pause
