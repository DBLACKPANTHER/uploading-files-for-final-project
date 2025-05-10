@echo off
echo ==========================
echo 🚀 Age Training Setup Start
echo ==========================

:: Activate venv
call "D:\Newcode\venv\Scripts\activate.bat"

:: Install required Python packages
echo Installing required Python packages...
pip install --upgrade pip
pip install tensorflow pandas matplotlib opencv-python scikit-learn kagglehub

:: Run the training script
echo ==========================
echo 🔁 Running NewTraining.py
echo ==========================
python NewTraining.py

echo ==========================
echo ✅ Done
pause
