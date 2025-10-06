# PowerShell script to run all three scripts simultaneously
Write-Host "Starting Live Target Detection Pipeline..." -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""

# Start the Live Confidence GUI in a new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python LiveConfidenceGUI.py" -WindowStyle Normal

# Wait a moment for GUI to initialize
Start-Sleep -Seconds 2

# Start operate.py in a new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python operate.py" -WindowStyle Normal

# Wait a moment for operate.py to initialize
Start-Sleep -Seconds 3

# Start TargetPoseEst01.py in a new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python TargetPoseEst01.py" -WindowStyle Normal

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "All three scripts are now running:" -ForegroundColor Yellow
Write-Host "- Live Confidence GUI: Monitor confidence values" -ForegroundColor Cyan
Write-Host "- Robot Operation: Control robot and detection" -ForegroundColor Cyan
Write-Host "- Target Pose Estimation: Analyze detections" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to close this launcher..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
