@echo off
echo Starting Live Target Detection Pipeline...
echo ==========================================
echo.

REM Start the Live Confidence GUI in a new window
start "Live Confidence GUI" cmd /k "python LiveConfidenceGUI.py"

REM Wait a moment for GUI to initialize
timeout /t 2 /nobreak >nul

REM Start operate.py in a new window
start "Robot Operation" cmd /k "python operate.py"

REM Wait a moment for operate.py to initialize
timeout /t 3 /nobreak >nul

REM Start TargetPoseEst01.py in a new window
start "Target Pose Estimation" cmd /k "python TargetPoseEst01.py"

echo.
echo ==========================================
echo All three scripts are now running:
echo - Live Confidence GUI: Monitor confidence values
echo - Robot Operation: Control robot and detection
echo - Target Pose Estimation: Analyze detections
echo.
echo Press any key to close this launcher...
pause >nul
