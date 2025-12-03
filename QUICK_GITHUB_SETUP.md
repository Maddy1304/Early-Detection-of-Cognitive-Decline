# Quick GitHub Setup - Step by Step

## Prerequisites Check
```powershell
# Check if Git is installed
git --version

# If not installed, download from: https://git-scm.com/download/win
```

## Step 1: Initialize Git (if not done)
```powershell
cd C:\Users\vaish\Desktop\project_1
git init
```

## Step 2: Verify .gitignore is Working
```powershell
# Check what will be ignored
git status --ignored

# You should see venv/, data/ravdess/, results/, *.log, etc. in the ignored list
```

## Step 3: Add Files
```powershell
# Add all files (respecting .gitignore)
git add .

# Check what will be committed
git status
```

## Step 4: First Commit
```powershell
git commit -m "Initial commit: Cognitive Decline Detection System

- Multi-modal federated learning implementation
- Edge-fog-cloud infrastructure simulation  
- Speech, facial, and gait analysis models
- RAVDESS dataset support
- Evaluation and visualization tools"
```

## Step 5: Create GitHub Repository

**Go to**: https://github.com/new

**Fill in**:
- Repository name: `cognitive-decline-detection` (or your choice)
- Description: `Early Detection of Cognitive Decline Using Multi-Modal Federated Learning`
- Public or Private: Choose based on your preference
- **DO NOT** check "Initialize with README" (you already have one)
- Click **Create repository**

## Step 6: Connect and Push

**After creating the repo, GitHub will show you commands. Use these:**

```powershell
# Replace YOUR_USERNAME with your actual GitHub username
git remote add origin https://github.com/YOUR_USERNAME/cognitive-decline-detection.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

**If asked for credentials**:
- Username: Your GitHub username
- Password: Use a **Personal Access Token** (not your password)
  - Create token: https://github.com/settings/tokens
  - Click "Generate new token (classic)"
  - Select scope: `repo` (full control)
  - Copy token and use it as password

## Step 7: Verify

Go to: `https://github.com/YOUR_USERNAME/cognitive-decline-detection`

**Check that**:
- ✅ Source code is there
- ✅ README.md is visible
- ✅ No `venv/` folder
- ✅ No `data/ravdess/` audio/video files
- ✅ No `results/` folder
- ✅ No `__pycache__/` folders

## Troubleshooting

### "Repository not found"
- Check username is correct
- Verify repo exists on GitHub

### "Authentication failed"
- Use Personal Access Token, not password
- Token must have `repo` scope

### "Large files detected"
- Check `.gitignore` is working: `git status --ignored`
- If files were already added:
  ```powershell
  git rm --cached <file>
  git commit -m "Remove large files"
  git push
  ```

## Future Updates

```powershell
git add .
git commit -m "Description of changes"
git push
```

Done! 🎉

