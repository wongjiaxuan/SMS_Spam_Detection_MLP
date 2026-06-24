@echo off
echo ============================================================
echo SMS Spam Detection System - MLP Classifier
echo ============================================================
echo.

:: Upgrade pip first
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

:: Install dependencies directly
echo Installing dependencies...
pip install pandas numpy scikit-learn nltk matplotlib seaborn joblib jupyter
echo.

:: Download NLTK data
echo Downloading NLTK data...
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('punkt_tab')"
echo.

:: Create directories
if not exist data mkdir data
if not exist models mkdir models
if not exist reports mkdir reports
if not exist reports\figures mkdir reports\figures

:: Run the main script
echo.
echo Running SMS Spam Detection System...
echo ============================================================
python main.py

echo.
echo ============================================================
echo Execution Complete!
echo ============================================================
pause