# GitHub Setup Guide

This guide will help you add your project to GitHub while excluding large datasets and unnecessary files.

## Prerequisites

1. **Git installed** - Check with: `git --version`
2. **GitHub account** - Create one at https://github.com if you don't have it
3. **GitHub CLI (optional)** - For easier authentication

## Step-by-Step Instructions

### Step 1: Initialize Git Repository (if not already done)

Open PowerShell in your project directory and run:

```powershell
# Check if git is already initialized
git status

# If not initialized, run:
git init
```

### Step 2: Check Current Status

```powershell
# See what files are currently tracked/untracked
git status
```

### Step 3: Verify .gitignore is Working

```powershell
# Check if .gitignore is excluding the right files
git status --ignored

# You should see:
# - venv/ (ignored)
# - data/ravdess/Audio_Speech_Actors_*/ (ignored)
# - results/ (ignored)
# - *.log files (ignored)
# - __pycache__/ (ignored)
```

### Step 4: Add Files to Git

```powershell
# Add all files (respecting .gitignore)
git add .

# Or add specific files/directories:
git add README.md
git add src/
git add config/
git add requirements.txt
git add setup.py
git add scripts/
git add .gitignore
```

### Step 5: Create Initial Commit

```powershell
# Create your first commit
git commit -m "Initial commit: Cognitive Decline Detection System

- Multi-modal federated learning implementation
- Edge-fog-cloud infrastructure simulation
- Speech, facial, and gait analysis models
- RAVDESS dataset support
- Evaluation and visualization tools"
```

### Step 6: Create GitHub Repository

**Option A: Using GitHub Website (Recommended for beginners)**

1. Go to https://github.com/new
2. Repository name: `cognitive-decline-detection` (or your preferred name)
3. Description: "Early Detection of Cognitive Decline Using Multi-Modal Federated Learning with Edge–Fog Collaboration"
4. Choose **Public** or **Private**
5. **DO NOT** initialize with README, .gitignore, or license (you already have these)
6. Click **Create repository**

**Option B: Using GitHub CLI**

```powershell
# Install GitHub CLI first: winget install GitHub.cli
# Then authenticate: gh auth login

# Create repository
gh repo create cognitive-decline-detection --public --description "Early Detection of Cognitive Decline Using Multi-Modal Federated Learning"
```

### Step 7: Connect Local Repository to GitHub

After creating the repository, GitHub will show you commands. Use these:

```powershell
# Add remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/cognitive-decline-detection.git

# Or if using SSH:
# git remote add origin git@github.com:YOUR_USERNAME/cognitive-decline-detection.git

# Verify remote was added
git remote -v
```

### Step 8: Push to GitHub

```powershell
# Push to GitHub (first time)
git branch -M main
git push -u origin main
```

If prompted for credentials:
- **Username**: Your GitHub username
- **Password**: Use a Personal Access Token (not your password)
  - Create token: https://github.com/settings/tokens
  - Select scopes: `repo` (full control of private repositories)

### Step 9: Verify Upload

1. Go to your repository on GitHub: `https://github.com/YOUR_USERNAME/cognitive-decline-detection`
2. Check that:
   - ✅ Source code is present
   - ✅ README.md is visible
   - ✅ No `venv/` folder
   - ✅ No `data/ravdess/` audio/video files
   - ✅ No `results/` folder with model files
   - ✅ No `__pycache__/` folders

## What Gets Excluded (Thanks to .gitignore)

The following will **NOT** be uploaded to GitHub:

- ✅ `venv/` - Virtual environment (too large)
- ✅ `data/ravdess/Audio_Speech_Actors_*/` - Audio dataset files
- ✅ `data/ravdess/Video_*/` - Video dataset files
- ✅ `results/` - Model checkpoints (.pth files), logs, plots
- ✅ `*.log` - All log files
- ✅ `__pycache__/` - Python cache files
- ✅ `*.pyc` - Compiled Python files
- ✅ `*.docx` - Review documents
- ✅ IDE configuration files

## What Gets Included

The following will be uploaded:

- ✅ All source code (`src/`)
- ✅ Configuration files (`config/`)
- ✅ Scripts (`scripts/`)
- ✅ Documentation (`README.md`, `*.md` files)
- ✅ Requirements (`requirements.txt`)
- ✅ Setup files (`setup.py`)
- ✅ Training scripts (`train.ps1`, `train.bat`)
- ✅ Tests (`tests/`)
- ✅ Empty data directory structure (with README files)

## Future Updates

After making changes to your code:

```powershell
# Check what changed
git status

# Add changes
git add .

# Commit with descriptive message
git commit -m "Description of your changes"

# Push to GitHub
git push
```

## Adding Dataset Download Instructions

Since datasets are excluded, add instructions in your README:

```markdown
## Dataset Setup

1. Download RAVDESS dataset from: [link]
2. Extract to `data/ravdess/`
3. Expected structure:
   ```
   data/ravdess/
   ├── Audio_Speech_Actors_01-24/
   ├── Video_Speech_Actors_01-24/
   └── README.txt
   ```
```

## Troubleshooting

### Issue: "Repository not found"
- Check your GitHub username is correct
- Verify repository exists on GitHub
- Check authentication (use Personal Access Token)

### Issue: "Large files detected"
- Check `.gitignore` is working: `git status --ignored`
- If large files were already committed:
  ```powershell
  git rm --cached <file>
  git commit -m "Remove large files"
  ```

### Issue: "Authentication failed"
- Use Personal Access Token instead of password
- Or set up SSH keys for easier access

## Best Practices

1. **Commit frequently** with descriptive messages
2. **Never commit**:
   - API keys or secrets
   - Large datasets
   - Model checkpoints
   - Virtual environments
3. **Always check** `git status` before committing
4. **Use branches** for new features:
   ```powershell
   git checkout -b feature-name
   git push -u origin feature-name
   ```

## Need Help?

- Git documentation: https://git-scm.com/doc
- GitHub Guides: https://guides.github.com
- Git ignore patterns: https://git-scm.com/docs/gitignore

