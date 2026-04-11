# GitHub Upload Guide

## Step 1 — Create the repo on GitHub

Go to https://github.com/new and create:
- Name: `smart-vehicle-twin`
- Description: `BMW i4 real-time digital twin — FastAPI · React · AWS ECS · Terraform · ML predictions · WebSocket`
- Visibility: Public
- Do NOT initialise with README (we have one)

## Step 2 — Push from this worktree

```bash
cd smart-vehicle-twin-main/.worktrees/svt-fullstack-mvp

git init                          # if not already a git repo
git add .
git commit -m "feat: complete Smart Vehicle Twin full-stack MVP"

git remote add origin https://github.com/YOUR_USERNAME/smart-vehicle-twin.git
git branch -M main
git push -u origin main
```

## Step 3 — Add GitHub Secrets (for CI/CD deploy)

Go to: Settings → Secrets and variables → Actions → New repository secret

| Secret | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | Your AWS IAM key |
| `AWS_SECRET_ACCESS_KEY` | Your AWS IAM secret |
| `AWS_REGION` | e.g. `eu-west-1` |
| `ECR_REPO` | e.g. `smart-vehicle-twin-backend` |
| `ECS_CLUSTER` | e.g. `svt-cluster` |
| `ECS_SERVICE` | e.g. `svt-backend` |
| `S3_BUCKET` | e.g. `svt-frontend-eu-west-1` |
| `CF_DIST_ID` | Your CloudFront distribution ID |

## Step 4 — Set repo topics (makes it discoverable)

Go to: repo page → ⚙ (gear icon next to About) → Topics

Add: `digital-twin` `fastapi` `react` `aws` `iot` `machine-learning` `websocket` `terraform` `python` `typescript` `ecs-fargate` `real-time`

## Step 5 — Pin the repo on your GitHub profile

Go to your profile → Customize your pins → select `smart-vehicle-twin`
