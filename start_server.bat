@echo off
echo ===================================================
echo     Starting Deepfake Detector AI Backend
echo ===================================================
echo.

echo 1/2 Checking and installing dependencies...
pip install -r requirements.txt

echo.
echo 2/2 Starting Flask Server...
python app.py

echo.
echo Server has stopped or encountered an error.
pause
