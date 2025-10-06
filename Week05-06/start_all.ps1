# Single command to start all three scripts
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python LiveConfidenceGUI.py" -WindowStyle Normal; Start-Process powershell -ArgumentList "-NoExit", "-Command", "python operate.py" -WindowStyle Normal; Start-Process powershell -ArgumentList "-NoExit", "-Command", "python TargetPoseEst01.py" -WindowStyle Normal
