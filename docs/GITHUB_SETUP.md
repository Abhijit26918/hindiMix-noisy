# GitHub Setup (One-time)

## Step 1: Create the repo on GitHub

1. Go to https://github.com/new
2. Repository name: `hindiMix-noisy`
3. Description: `Noise-robust hate speech detection in Hindi-English code-mixed text`
4. Set to **Public** (needed for citations and paper links)
5. Do NOT initialize with README (we already have one)
6. Click **Create repository**

## Step 2: Link your local repo to GitHub

Run these commands in the project folder:

```bash
cd "d:/multilingual hate speech _code mixed/hindiMix-noisy"

# Add remote
git remote add origin https://github.com/Abhijit26918/hindiMix-noisy.git

# Push for the first time
git branch -M master
git push -u origin master
```

## Step 3: Daily push routine (every evening)

```bash
cd "d:/multilingual hate speech _code mixed/hindiMix-noisy"
bash daily_push.sh "What you did today"
```

Example messages:
- `bash daily_push.sh "Phase 1: Downloaded HASOC and SemEval datasets"`
- `bash daily_push.sh "Phase 1: Noise generation complete, 30K noisy samples"`
- `bash daily_push.sh "Phase 2: MuRIL baseline F1=0.72 on clean test"`
