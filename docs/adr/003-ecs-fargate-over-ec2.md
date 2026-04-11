# ADR 003 — ECS Fargate over EC2 for backend deployment

**Date:** 2026-04-10  
**Status:** Accepted

## Context

The backend is a single stateless FastAPI container. It needs to be deployed to AWS with minimal operational overhead.

## Decision

Use **ECS Fargate** (256 CPU / 512 MB) behind an **Application Load Balancer**.

## Reasons

- No EC2 instance management — no patching, no AMI selection, no SSH keys
- Pay-per-task — cost-efficient for a demo/portfolio workload that may be idle
- ALB handles WebSocket upgrade transparently on the same listener as REST
- Fargate integrates natively with ECR, CloudWatch Logs, and IAM task roles
- Terraform resource is ~30 lines vs managing an Auto Scaling Group

## Consequences

- Cold start latency (~10–15 s) on first request after scale-to-zero — acceptable for demo
- No persistent local storage — models are baked into the image or mounted via EFS (future)
