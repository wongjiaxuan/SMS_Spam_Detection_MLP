@echo off
:: ============================================================
:: SMS Spam Detection System - Setup Script (Windows)
:: ============================================================

echo ============================================================
echo   SMS SPAM DETECTION SYSTEM - VIRTUAL ENVIRONMENT SETUP
echo ============================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed. Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

python --version
echo.

:: Create virtual environment
echo Creating virtual environment...
if exist venv (
    echo Virtual environment already exists. Removing old one...
    rmdir /s /q venv
)

python -m venv venv
echo Virtual environment created successfully!
echo.

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

:: Install requirements
echo Installing dependencies...
pip install -r requirements.txt
echo.

:: Download NLTK data
echo Downloading NLTK data...
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('punkt_tab')"
echo.

:: Create necessary directories
echo Creating required directories...
if not exist data mkdir data
if not exist models mkdir models
if not exist reports mkdir reports
if not exist reports\figures mkdir reports\figures
echo.

:: Check if dataset exists
if not exist data\spam.csv (
    echo ============================================================
    echo WARNING: Dataset not found in 'data\spam.csv'
    echo Please download the SMS Spam Collection dataset from:
    echo   - UCI: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
    echo   - Kaggle: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
    echo ============================================================
)

echo ============================================================
echo   SETUP COMPLETE!
echo ============================================================
echo.
echo To activate the virtual environment, run:
echo   venv\Scripts\activate
echo.
echo To run the project, use:
echo   python main.py
echo.
echo Or use the run script:
echo   run.bat
echo ============================================================
pause