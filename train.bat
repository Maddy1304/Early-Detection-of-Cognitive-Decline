@echo off
REM Training Script for Cognitive Decline Detection Project
REM Batch script version for Windows Command Prompt

echo ========================================
echo   Training Script
echo ========================================
echo.

REM Create results directories
if not exist "results\demo\speech" mkdir "results\demo\speech"
if not exist "results\demo\facial" mkdir "results\demo\facial"
if not exist "results\demo\multimodal" mkdir "results\demo\multimodal"
if not exist "results\demo\evaluation" mkdir "results\demo\evaluation"
if not exist "results\demo\simulation" mkdir "results\demo\simulation"

echo === Part 1: Model Training ===
echo.

echo Training Speech Model...
python src/main.py --mode training --dataset ravdess --model speech --output results/demo/speech
if errorlevel 1 (
    echo Speech model training failed!
    exit /b 1
)
echo Speech model training completed!
echo.

echo Training Facial Model...
python src/main.py --mode training --dataset ravdess --model facial --output results/demo/facial
if errorlevel 1 (
    echo Facial model training failed!
    exit /b 1
)
echo Facial model training completed!
echo.

echo Training Multimodal Model...
python src/main.py --mode training --dataset ravdess --model multimodal --output results/demo/multimodal
if errorlevel 1 (
    echo Multimodal model training failed!
    exit /b 1
)
echo Multimodal model training completed!
echo.

echo === Part 2: Infrastructure Simulation ===
echo.
echo Running Simulation (10 minutes)...
python src/main.py --mode simulation --dataset ravdess --model multimodal --output results/demo/simulation --duration 600
if errorlevel 1 (
    echo Simulation failed!
    exit /b 1
)
echo Simulation completed!
echo.

echo === Part 3: Model Evaluation ===
echo.
echo Evaluating Models...
python src/main.py --mode evaluation --dataset ravdess --model multimodal --output results/demo/evaluation
if errorlevel 1 (
    echo Evaluation failed!
    exit /b 1
)
echo Evaluation completed!
echo.

echo ========================================
echo   Demo Complete!
echo ========================================
echo Check results\demo\ directory for outputs
echo.
pause

