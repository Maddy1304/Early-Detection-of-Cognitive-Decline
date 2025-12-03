# Training Script for Cognitive Decline Detection Project
# This script trains speech, facial, and multimodal models on RAVDESS dataset

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Training Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Ensure results directories exist
New-Item -ItemType Directory -Force -Path "results/demo/speech" | Out-Null
New-Item -ItemType Directory -Force -Path "results/demo/facial" | Out-Null
New-Item -ItemType Directory -Force -Path "results/demo/multimodal" | Out-Null
New-Item -ItemType Directory -Force -Path "results/demo/evaluation" | Out-Null
New-Item -ItemType Directory -Force -Path "results/demo/simulation" | Out-Null

Write-Host "=== Part 1: Model Training ===" -ForegroundColor Green
Write-Host ""

Write-Host "Training Speech Model..." -ForegroundColor Yellow
python src/main.py --mode training --dataset ravdess --model speech --output results/demo/speech

if ($LASTEXITCODE -eq 0) {
    Write-Host "Speech model training completed!" -ForegroundColor Green
} else {
    Write-Host "Speech model training failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Training Facial Model..." -ForegroundColor Yellow
python src/main.py --mode training --dataset ravdess --model facial --output results/demo/facial

if ($LASTEXITCODE -eq 0) {
    Write-Host "Facial model training completed!" -ForegroundColor Green
} else {
    Write-Host "Facial model training failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Training Multimodal Model..." -ForegroundColor Yellow
python src/main.py --mode training --dataset ravdess --model multimodal --output results/demo/multimodal

if ($LASTEXITCODE -eq 0) {
    Write-Host "Multimodal model training completed!" -ForegroundColor Green
} else {
    Write-Host "Multimodal model training failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== Part 2: Infrastructure Simulation ===" -ForegroundColor Green
Write-Host ""
Write-Host "Running Simulation (10 minutes)..." -ForegroundColor Yellow
python src/main.py --mode simulation --dataset ravdess --model multimodal --output results/demo/simulation --duration 600

if ($LASTEXITCODE -eq 0) {
    Write-Host "Simulation completed!" -ForegroundColor Green
} else {
    Write-Host "Simulation failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== Part 3: Model Evaluation ===" -ForegroundColor Green
Write-Host ""
Write-Host "Evaluating Models..." -ForegroundColor Yellow
python src/main.py --mode evaluation --dataset ravdess --model multimodal --output results/demo/evaluation

if ($LASTEXITCODE -eq 0) {
    Write-Host "Evaluation completed!" -ForegroundColor Green
} else {
    Write-Host "Evaluation failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Demo Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Check results/demo/ directory for outputs" -ForegroundColor Yellow
Write-Host ""

