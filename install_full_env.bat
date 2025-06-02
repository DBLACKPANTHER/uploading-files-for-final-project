@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

:: === הגדרות ===
set PYTHON_VERSION=3.11.5
set PYTHON_DIR=%LOCALAPPDATA%\Programs\Python\Python311
set PYTHON_EXE=%PYTHON_DIR%\python.exe
set PROJECT_DIR=%CD%
set VENV_DIR=%PROJECT_DIR%\venv

:: === הורדת פייתון אם לא קיים ===
if not exist "%PYTHON_EXE%" (
    echo Python %PYTHON_VERSION% not found. Downloading...
    curl -L -o python-installer.exe https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-amd64.exe
    echo Installing Python %PYTHON_VERSION%...
    start /wait python-installer.exe /quiet InstallAllUsers=0 PrependPath=1 Include_test=0 TargetDir="%PYTHON_DIR%"
    del python-installer.exe
) else (
    echo Python %PYTHON_VERSION% already installed.
)

:: === יצירת סביבת עבודה חדשה ===
echo Creating virtual environment...
rmdir /s /q "%VENV_DIR%" 2>nul
"%PYTHON_EXE%" -m venv "%VENV_DIR%"
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Failed to create virtual environment.
    pause
    exit /b
)

:: === הפעלת הסביבה והתקנת pip ===
call "%VENV_DIR%\Scripts\activate.bat"
echo Upgrading pip...
python -m pip install --upgrade pip --no-cache-dir

:: === יצירת קובץ requirements.txt ===
echo Creating requirements.txt...
> requirements.txt (
    echo absl-py==2.2.2
    echo astunparse==1.6.3
    echo beautifulsoup4==4.13.4
    echo blinker==1.9.0
    echo cachetools==5.5.2
    echo certifi==2025.4.26
    echo charset-normalizer==3.4.2
    echo click==8.1.8
    echo colorama==0.4.6
    echo contourpy==1.3.2
    echo cycler==0.12.1
    echo filelock==3.18.0
    echo Flask==3.1.0
    echo flatbuffers==25.2.10
    echo fonttools==4.58.0
    echo gast==0.6.0
    echo gdown==5.2.0
    echo google-auth==2.40.1
    echo google-auth-oauthlib==1.0.0
    echo google-pasta==0.2.0
    echo grpcio==1.71.0
    echo h5py==3.13.0
    echo idna==3.10
    echo itsdangerous==2.2.0
    echo Jinja2==3.1.6
    echo joblib==1.5.0
    echo kagglehub==0.3.12
    echo keras==2.14.0
    echo kiwisolver==1.4.8
    echo libclang==18.1.1
    echo Markdown==3.8
    echo MarkupSafe==3.0.2
    echo matplotlib==3.10.3
    echo ml-dtypes==0.2.0
    echo numpy==1.26.4
    echo oauthlib==3.2.2
    echo opencv-python==4.11.0.86
    echo opt_einsum==3.4.0
    echo packaging==25.0
    echo pandas==2.2.3
    echo pillow==11.2.1
    echo protobuf==4.25.7
    echo pyasn1==0.6.1
    echo pyasn1_modules==0.4.2
    echo pyparsing==3.2.3
    echo PySocks==1.7.1
    echo python-dateutil==2.9.0.post0
    echo pytz==2025.2
    echo PyYAML==6.0.2
    echo requests==2.32.3
    echo requests-oauthlib==2.0.0
    echo retina-face==0.0.17
    echo rsa==4.9.1
    echo scikit-learn==1.6.1
    echo scipy==1.15.3
    echo setuptools==65.5.0
    echo six==1.17.0
    echo soupsieve==2.7
    echo tensorboard==2.14.1
    echo tensorboard-data-server==0.7.2
    echo tensorflow==2.14.0
    echo tensorflow-estimator==2.14.0
    echo tensorflow-intel==2.14.0
    echo tensorflow-io-gcs-filesystem==0.31.0
    echo termcolor==3.1.0
    echo threadpoolctl==3.6.0
    echo tqdm==4.67.1
    echo typing_extensions==4.13.2
    echo tzdata==2025.2
    echo urllib3==2.4.0
    echo Werkzeug==3.1.3
    echo wheel==0.45.1
    echo wrapt==1.14.1
)

:: === התקנת חבילות ===
echo Installing packages...
pip install -r requirements.txt --no-cache-dir

if %errorlevel% neq 0 (
    echo Package installation failed.
    pause
    exit /b 1
)

:: === פתיחת Visual Studio Code ===
echo Opening Visual Studio Code...
code .

echo In VS Code, choose this interpreter:
echo   %VENV_DIR%\Scripts\python.exe

ENDLOCAL
pause
