@echo off
echo Starting all three scripts...
start "GUI" python LiveConfidenceGUI.py
start "Robot" python operate.py  
start "Analysis" python TargetPoseEst01.py
echo All scripts started!
