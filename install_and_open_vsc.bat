@echo off
SETLOCAL

cd /d D:\Newcode

echo 🔄 Creating a clean virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo  Failed to create venv. Make sure Python is installed and on PATH.
    pause
    exit /b 1
)

echo  Virtual environment created.
pause

echo  Activating virtual environment...
call venv\Scripts\activate.bat

echo  Upgrading pip...
python -m pip install --upgrade pip --no-cache-dir
pause

echo  Installing required packages one-by-one...
pip install tensorflow==2.14.0 --no-cache-dir
pip install opencv-python --no-cache-dir
pip install pillow flask --no-cache-dir
pip install retina-face --no-cache-dir
pip install kagglehub pandas matplotlib scikit-learn --no-cache-dir

if %errorlevel% neq 0 (
    echo  Package installation failed.
    pause
    exit /b 1
)

echo  All packages installed successfully.
pause

echo  Opening Visual Studio Code...
code .

echo  In VS Code, go to "Python: Select Interpreter" and choose:
echo    D:\Newcode\venv\Scripts\python.exe
echo Then run FlaskApp.py or NewTraining.py manually.

ENDLOCAL
pause
