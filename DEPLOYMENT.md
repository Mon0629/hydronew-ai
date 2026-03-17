# Railway Deployment Guide

This guide covers deploying the Hydronew AI classification service to Railway.

## Prerequisites

- Railway account ([railway.app](https://railway.app))
- GitHub repository with this project
- Git LFS installed locally (for migration step)

## Step 1: Migrate LFS Files (Required)

Railway does **not** pull Git LFS files. Your `data/` and `models/` paths were previously tracked with LFS. These have been removed from `.gitattributes`, but existing files in the repo are still LFS pointers.

Before deploying, convert them to regular files:

```bash
# Migrate data and models out of LFS into regular Git history
git lfs migrate export --include="data/*,models/*" --everything

# Commit the migration (rewrites history)
git add . && git commit -m "Migrate data and models out of LFS for Railway deployment"
git push --force-with-lease
```

**Note:** This rewrites Git history. If others use this repo, coordinate with them. Alternatively, run `git lfs pull` locally, then remove the LFS attributes and recommit the files (without `--everything` if you want to avoid rewriting history).

## Step 2: Create Railway Project

1. Go to [railway.app](https://railway.app) and sign in.
2. Click **New Project** → **Deploy from GitHub repo**.
3. Select your `hydronew-ai` repository.
4. Railway will detect the Dockerfile and build the image.

## Step 3: Configure Environment Variables

In your Railway service, go to **Variables** and add:

### MQTT (required for classification)

| Variable          | Description                    | Example                              |
|-------------------|--------------------------------|--------------------------------------|
| `MQTT_BROKER`     | HiveMQ Cloud broker host       | `xxx.s1.eu.hivemq.cloud`             |
| `MQTT_PORT`       | Port (use 8883 for TLS)        | `8883`                               |
| `MQTT_USERNAME`   | MQTT username                  | `your_username`                      |
| `MQTT_PASSWORD`   | MQTT password                  | `your_password`                      |
| `MQTT_CLIENT_ID`  | Optional client ID            | `hydronew_ai_classification`         |

### Database (optional – service runs without it)

| Variable      | Description   | Example   |
|---------------|---------------|-----------|
| `DB_HOST`     | MySQL host    | `...`    |
| `DB_PORT`     | Port          | `3306`   |
| `DB_USER`     | Username      | `root`   |
| `DB_PASSWORD` | Password      | `...`    |
| `DB_NAME`     | Database name | `hydronew` |

## Step 4: Build and Deploy

- Railway uses the `Dockerfile` in the project root.
- The image trains the model during build (`python run_pipeline.py train`) and runs the classification service at startup (`python run_pipeline.py classify`).
- The start command is defined in `railway.json` but can be overridden in the service settings.

## Troubleshooting

### KeyError: 118 when loading model

This usually means:

1. **LFS pointers:** The model in the repo is an LFS pointer. Run the LFS migration in Step 1.
2. **Version mismatch:** The model was trained with different scikit-learn/joblib versions. The Dockerfile trains the model during build, so it should match the runtime versions.

### Build fails during `python run_pipeline.py train`

- Ensure `data/raw/water_data_quality.csv` exists and is **not** an LFS pointer after migration.
- Check build logs for missing files or import errors.

### MQTT connection timeout

- Verify `MQTT_BROKER`, `MQTT_PORT`, `MQTT_USERNAME`, and `MQTT_PASSWORD` in Railway variables.
- Ensure your HiveMQ Cloud project allows connections from Railway’s IP ranges.
