# Training Commands for Cognitive Decline Detection Project

## Quick Start

### Option 1: Use the Automated Scripts (Recommended)

**Windows PowerShell:**
```powershell
.\train.ps1
```

**Windows Command Prompt:**
```cmd
train.bat
```

### Option 2: Run Commands Manually

## Individual Training Commands

### 1. Train Speech Model

```powershell
# PowerShell / Command Prompt
python src/main.py --mode training --dataset ravdess --model speech --output results/demo/speech
```

```bash
# Linux/Mac
python3 src/main.py --mode training --dataset ravdess --model speech --output results/demo/speech
```

### 2. Train Facial Model

```powershell
python src/main.py --mode training --dataset ravdess --model facial --output results/demo/facial
```

### 3. Train Multimodal Model

```powershell
python src/main.py --mode training --dataset ravdess --model multimodal --output results/demo/multimodal
```

### 4. Run Infrastructure Simulation

```powershell
python src/main.py --mode simulation --dataset ravdess --model multimodal --output results/demo/simulation --duration 600
```

### 5. Evaluate Models

```powershell
python src/main.py --mode evaluation --dataset ravdess --model multimodal --output results/demo/evaluation
```

## Complete Training Workflow

### Step-by-Step Commands

**Step 1: Activate Virtual Environment**
```powershell
# PowerShell
.\venv\Scripts\Activate.ps1

# Command Prompt
venv\Scripts\activate.bat
```

**Step 2: Train All Models**

Run these commands in sequence:

```powershell
# Train Speech Model
python src/main.py --mode training --dataset ravdess --model speech --output results/demo/speech

# Train Facial Model  
python src/main.py --mode training --dataset ravdess --model facial --output results/demo/facial

# Train Multimodal Model
python src/main.py --mode training --dataset ravdess --model multimodal --output results/demo/multimodal
```

**Step 3: Run Simulation (Optional)**

```powershell
python src/main.py --mode simulation --dataset ravdess --model multimodal --output results/demo/simulation --duration 600
```

**Step 4: Evaluate Models**

```powershell
python src/main.py --mode evaluation --dataset ravdess --model multimodal --output results/demo/evaluation
```

## Command-Line Arguments Reference

### Available Modes
- `training` - Train a model on the dataset
- `evaluation` - Evaluate a trained model
- `simulation` - Run infrastructure simulation
- `demo` - Run complete demonstration

### Available Datasets
- `ravdess` - RAVDESS Audio-Visual Database
- `daic-woz` - DAIC-WOZ Depression Dataset
- `mpower` - mPower Parkinson's Dataset
- `all` - Use all datasets

### Available Models
- `speech` - Speech/audio analysis model
- `facial` - Facial expression model
- `gait` - Gait analysis model
- `multimodal` - Multimodal fusion model

### Common Arguments

```
--mode          : Operation mode (training/evaluation/simulation/demo)
--dataset       : Dataset to use (ravdess/daic-woz/mpower/all)
--model         : Model type (speech/facial/gait/multimodal)
--output        : Output directory for results
--config        : Path to config file (default: config/simulation_config.yaml)
--duration      : Simulation duration in seconds (default: 3600)
--log-level     : Logging level (DEBUG/INFO/WARNING/ERROR)
--privacy       : Enable privacy-preserving techniques
--visualize     : Generate visualization plots
```

## Examples

### Train Only Speech Model with Custom Output

```powershell
python src/main.py --mode training --dataset ravdess --model speech --output results/speech_only --log-level DEBUG
```

### Train Multimodal Model with Visualization

```powershell
python src/main.py --mode training --dataset ravdess --model multimodal --output results/multimodal_viz --visualize
```

### Quick Test Run (Fewer Epochs)

First, modify `config/model_config.yaml` to reduce epochs:
```yaml
training:
  epochs: 2  # Reduced from 10 for quick testing
```

Then run:
```powershell
python src/main.py --mode training --dataset ravdess --model speech --output results/quick_test
```

### Evaluate Specific Trained Model

```powershell
# Ensure model exists in output directory first
python src/main.py --mode evaluation --dataset ravdess --model speech --output results/demo/speech
```

## Expected Output Locations

After training, check these directories:

- `results/demo/speech/`
  - `best_model.pth` - Trained model weights
  - `training_results.json` - Training metrics
  - `logs/` - Training logs

- `results/demo/facial/`
  - Same structure as speech

- `results/demo/multimodal/`
  - Same structure as speech

- `results/demo/evaluation/`
  - `evaluation_results.json` - Evaluation metrics
  - `logs/` - Evaluation logs

- `results/demo/simulation/`
  - `simulation_data.json` - Simulation data
  - `simulation_report.json` - Simulation report

## Troubleshooting

### Common Issues

1. **Dataset Not Found**
   ```
   Error: Dataset not found at data/ravdess
   ```
   - Ensure RAVDESS dataset is downloaded and extracted to `data/ravdess/`

2. **CUDA Out of Memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   - Reduce batch size in `config/model_config.yaml`:
     ```yaml
     training:
       batch_size: 16  # Reduce from 32
     ```

3. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'src'
   ```
   - Ensure you're running from project root directory
   - Ensure virtual environment is activated

4. **Model Not Found for Evaluation**
   ```
   FileNotFoundError: Trained model not found
   ```
   - Train the model first using `--mode training`
   - Check that `best_model.pth` exists in output directory

### Performance Tips

1. **Use GPU if Available**
   - PyTorch will automatically use CUDA if available
   - Check with: `python -c "import torch; print(torch.cuda.is_available())"`

2. **Adjust Batch Size**
   - Larger batch size = faster training but more memory
   - Smaller batch size = slower but less memory

3. **Reduce Epochs for Testing**
   - Change `epochs: 10` to `epochs: 2` in config for quick testing

## Monitoring Training

Training progress is shown with progress bars and logged to:
- Console output (real-time)
- Log files in `results/*/logs/`
- JSON results in `results/*/training_results.json`

## Next Steps After Training

1. Check training results in JSON files
2. Visualize metrics (if `--visualize` flag used)
3. Evaluate models on test set
4. Run simulations to test infrastructure
5. Compare model performances

