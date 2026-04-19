# Step-by-Step AWS Deployment Guide

**Account:** `685057748560` | **Region:** `us-east-1` | **Parser:** Ollama (local, no Z.AI key)

This guide is the single source of truth for deploying the MultiModal RAG pipeline to AWS from a
completely fresh account. Follow the phases in order — each phase depends on the one before it.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [IAM — Create Admin User](#2-iam--create-admin-user)
3. [Shell Variables](#3-shell-variables)
4. [Security Groups](#4-security-groups)
5. [ECR Repositories](#5-ecr-repositories)
6. [ECS Cluster](#6-ecs-cluster)
7. [EFS — Persistent Storage](#7-efs--persistent-storage)
8. [Secrets Manager](#8-secrets-manager)
9. [IAM — CI/CD Bot User](#9-iam--cicd-bot-user)
10. [IAM — ECS Task Execution Role](#10-iam--ecs-task-execution-role)
11. [CloudWatch Log Groups](#11-cloudwatch-log-groups)
12. [ECS Task Definitions](#12-ecs-task-definitions)
13. [Application Load Balancer](#13-application-load-balancer)
14. [ECS Services](#14-ecs-services)
15. [Ollama Model Bootstrap](#15-ollama-model-bootstrap)
16. [GitHub Actions Secrets](#16-github-actions-secrets)
17. [Verify Deployment](#17-verify-deployment)
18. [Troubleshooting Common Deployment Issues](#18-troubleshooting-common-deployment-issues)
19. [CI/CD Flow Reference](#19-cicd-flow-reference)
20. [Rollback Procedure](#20-rollback-procedure)
21. [Cost Overview](#21-cost-overview--what-this-infrastructure-charges-per-month)
22. [How to Stop the Infrastructure (Save Money)](#22-how-to-stop-the-infrastructure-save-money-keep-data)
23. [How to Restart the Infrastructure](#23-how-to-restart-the-infrastructure)
24. [How to Tear Down Everything (Full Deletion)](#24-how-to-tear-down-everything-full-deletion)

---

## 1. Prerequisites

### 1.1 AWS CLI v2

```bash
aws --version
# Expected: aws-cli/2.x.x
```

If not installed: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

### 1.2 Docker

```bash
docker --version
# Expected: Docker version 24.x or higher
```

### 1.3 jq

```bash
brew install jq       # macOS
# apt-get install jq  # Ubuntu/Debian
```

### 1.4 GitHub CLI (for setting secrets later)

```bash
brew install gh
gh auth login
```

---

## 2. IAM — Create Admin User

This step is done **once via the AWS Console** using your root account. After this, you never use
root credentials again.

### 2.1 Create the user

1. AWS Console → **IAM** → **Users** → **Create user**
2. **User name:** `doc-parser-admin`
3. Do **not** enable console access (CLI only)
4. Click **Next**

### 2.2 Attach permissions

Choose **"Attach policies directly"** → check **`AdministratorAccess`**

> `AdministratorAccess` is used here because the setup phase touches EC2, ECS, ECR, EFS, ALB,
> Secrets Manager, IAM, and CloudWatch. The CI/CD bot user (`doc-parser-cicd`) created in
> Phase 9 is tightly scoped — that is the credential that matters for ongoing security.

### 2.3 Generate access keys

1. Click through to **Create user**
2. Open the user → **Security credentials** tab → **Create access key**
3. Choose **"Command Line Interface (CLI)"** → confirm → **Create access key**
4. **Save the Access Key ID and Secret Access Key** — they are shown only once

### 2.4 Configure AWS CLI

```bash
aws configure --profile doc-parser-admin
# AWS Access Key ID:     <paste key id>
# AWS Secret Access Key: <paste secret key>
# Default region name:   us-east-1
# Default output format: json
```

### 2.5 Verify

```bash
aws sts get-caller-identity --profile doc-parser-admin
# Expected: "Arn": "arn:aws:iam::685057748560:user/doc-parser-admin"
```

### 2.6 Export profile for the session

```bash
export AWS_PROFILE=doc-parser-admin
```

All subsequent commands in this guide use this profile automatically.

---

## 3. Shell Variables

Run these at the start of every terminal session before executing any commands in this guide.
Replace the VPC_ID and SUBNET_IDS placeholders with your actual values (retrieved below).

### 3.1 Retrieve your default VPC and subnets

```bash
# Get default VPC ID
aws ec2 describe-vpcs \
  --filters "Name=isDefault,Values=true" \
  --query 'Vpcs[0].VpcId' \
  --output text \
  --region us-east-1

# List all default subnets (pick 2 from different AZs)
aws ec2 describe-subnets \
  --filters "Name=defaultForAz,Values=true" \
  --query 'Subnets[*].{SubnetId:SubnetId,AZ:AvailabilityZone}' \
  --output table \
  --region us-east-1
```

### 3.2 Export all variables

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=685057748560
export ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
export CLUSTER_NAME=doc-parser-cluster

# Replace with your actual IDs from the commands above
export VPC_ID=vpc-XXXXXXXXXXXXXXXXX
export SUBNET_IDS=subnet-XXXXXXXXXXXXXXXXX,subnet-XXXXXXXXXXXXXXXXX
```

### 3.3 Verify

```bash
echo "Account : $AWS_ACCOUNT_ID"
echo "Region  : $AWS_REGION"
echo "VPC     : $VPC_ID"
echo "Subnets : $SUBNET_IDS"
echo "ECR     : $ECR_REGISTRY"
```

All five lines must be non-empty before proceeding.

---

## 4. Security Groups

Two security groups are needed:
- **ALB SG** — faces the internet, accepts port 80
- **ECS SG** — faces the ALB only, accepts port 8000

```bash
# --- ALB Security Group ---
ALB_SG=$(aws ec2 create-security-group \
  --group-name doc-parser-alb-sg \
  --description "ALB for doc-parser" \
  --vpc-id $VPC_ID \
  --query 'GroupId' --output text)
echo "ALB SG: $ALB_SG"

# Allow HTTP from internet
aws ec2 authorize-security-group-ingress \
  --group-id $ALB_SG \
  --protocol tcp --port 80 --cidr 0.0.0.0/0

# --- ECS Security Group ---
ECS_SG=$(aws ec2 create-security-group \
  --group-name doc-parser-ecs-sg \
  --description "ECS tasks for doc-parser" \
  --vpc-id $VPC_ID \
  --query 'GroupId' --output text)
echo "ECS SG: $ECS_SG"

# Allow ALB → ECS on FastAPI port
aws ec2 authorize-security-group-ingress \
  --group-id $ECS_SG \
  --protocol tcp --port 8000 --source-group $ALB_SG

# Allow EFS mount traffic within ECS tasks
aws ec2 authorize-security-group-ingress \
  --group-id $ECS_SG \
  --protocol tcp --port 2049 --source-group $ECS_SG
```

```bash
# ✅ Verify both rules were applied — you should see port 8000 and port 2049
aws ec2 describe-security-groups \
  --group-ids $ECS_SG \
  --query 'SecurityGroups[0].IpPermissions[*].{port:FromPort,source:UserIdGroupPairs[0].GroupId}' \
  --output table
```

> Expected: one row for port `8000` (source = ALB SG id) and one row for port `2049` (source = ECS SG id). If port 8000 is missing, the ALB will never reach your containers and health checks will time out with `Target.Timeout`.

> **Save these values** — you will need them in later phases.

---

## 5. ECR Repositories

Container images are stored in ECR. One repository for the FastAPI app.

```bash
aws ecr create-repository \
  --repository-name doc-parser/app \
  --region $AWS_REGION \
  --image-scanning-configuration scanOnPush=true
```

Verify:

```bash
aws ecr describe-repositories \
  --query 'repositories[*].repositoryName' \
  --output table \
  --region $AWS_REGION
```

---

## 6. ECS Cluster

```bash

aws iam create-service-linked-role --aws-service-name ecs.amazonaws.com


aws ecs create-cluster \
  --cluster-name $CLUSTER_NAME \
  --capacity-providers FARGATE FARGATE_SPOT \
  --region $AWS_REGION
```

Verify:

```bash
aws ecs describe-clusters \
  --clusters $CLUSTER_NAME \
  --query 'clusters[0].{name:clusterName,status:status}' \
  --output table
```

---

## 7. EFS — Persistent Storage

EFS provides two persistent volumes that survive deployments:
- `/qdrant/storage` — Qdrant vector database data
- `/root/.ollama` — Ollama model weights (downloaded once, reused forever)

```bash
# Create the file system
FS_ID=$(aws efs create-file-system \
  --performance-mode generalPurpose \
  --throughput-mode bursting \
  --region $AWS_REGION \
  --query 'FileSystemId' --output text)
echo "EFS ID: $FS_ID"

# Wait until available (check lifecycle state)
aws efs describe-file-systems \
  --file-system-id $FS_ID \
  --query 'FileSystems[0].LifeCycleState' \
  --output text
# Wait until output is: available

# Create mount targets — one per subnet (repeat for each subnet in SUBNET_IDS)
SUBNET1=$(echo $SUBNET_IDS | cut -d',' -f1)
SUBNET2=$(echo $SUBNET_IDS | cut -d',' -f2)

aws efs create-mount-target \
  --file-system-id $FS_ID \
  --subnet-id $SUBNET1 \
  --security-groups $ECS_SG

aws efs create-mount-target \
  --file-system-id $FS_ID \
  --subnet-id $SUBNET2 \
  --security-groups $ECS_SG

# Access point for Qdrant data
QDRANT_AP=$(aws efs create-access-point \
  --file-system-id $FS_ID \
  --posix-user Uid=1000,Gid=1000 \
  --root-directory "Path=/qdrant,CreationInfo={OwnerUid=1000,OwnerGid=1000,Permissions=755}" \
  --query 'AccessPointId' --output text)
echo "Qdrant Access Point: $QDRANT_AP"

# Access point for Ollama model weights
OLLAMA_AP=$(aws efs create-access-point \
  --file-system-id $FS_ID \
  --posix-user Uid=0,Gid=0 \
  --root-directory "Path=/ollama,CreationInfo={OwnerUid=0,OwnerGid=0,Permissions=755}" \
  --query 'AccessPointId' --output text)
echo "Ollama Access Point: $OLLAMA_AP"
```

> **Save `FS_ID`, `QDRANT_AP`, and `OLLAMA_AP`** — required for task definitions in Phase 12.

---

## 8. Secrets Manager

Only `OPENAI_API_KEY` is needed. This project uses Ollama locally — no Z.AI API key required.

```bash
aws secretsmanager create-secret \
  --name doc-parser/openai-api-key \
  --secret-string '{"OPENAI_API_KEY":"sk-...YOUR-KEY-HERE..."}' \
  --region $AWS_REGION
```

To update the key later:

```bash
aws secretsmanager put-secret-value \
  --secret-id doc-parser/openai-api-key \
  --secret-string '{"OPENAI_API_KEY":"sk-...NEW-KEY..."}'
```

---

## 9. IAM — CI/CD Bot User

This is the machine user whose credentials go into GitHub Actions. It can push images to ECR and
trigger ECS deployments — nothing else.

```bash
# Create user
aws iam create-user --user-name doc-parser-cicd

# Create access key — SAVE the output (shown only once)
aws iam create-access-key --user-name doc-parser-cicd
```

> Copy the `AccessKeyId` and `SecretAccessKey` from the output. You will use these in Phase 16
> (GitHub Actions secrets).

```bash
# Create scoped policy
cat > /tmp/cicd-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ECRAuth",
      "Effect": "Allow",
      "Action": ["ecr:GetAuthorizationToken"],
      "Resource": "*"
    },
    {
      "Sid": "ECRPush",
      "Effect": "Allow",
      "Action": [
        "ecr:BatchCheckLayerAvailability",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "ecr:PutImage",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage"
      ],
      "Resource": [
        "arn:aws:ecr:*:*:repository/doc-parser/app"
      ]
    },
    {
      "Sid": "ECSDeployServices",
      "Effect": "Allow",
      "Action": ["ecs:UpdateService", "ecs:DescribeServices"],
      "Resource": "*"
    },
    {
      "Sid": "ECSWaitForStable",
      "Effect": "Allow",
      "Action": [
        "ecs:DescribeTaskDefinition",
        "ecs:ListTasks",
        "ecs:DescribeTasks"
      ],
      "Resource": "*"
    }
  ]
}
EOF

aws iam put-user-policy \
  --user-name doc-parser-cicd \
  --policy-name doc-parser-cicd-policy \
  --policy-document file:///tmp/cicd-policy.json
```

---

## 10. IAM — ECS Task Execution Role

This is an IAM **Role** (not a user) that Fargate assumes at runtime to pull images, write logs,
read secrets, and mount EFS volumes.

```bash
# Create the role
aws iam create-role \
  --role-name doc-parser-ecs-task-execution \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "ecs-tasks.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# AWS-managed policy: ECR pull + CloudWatch logs
aws iam attach-role-policy \
  --role-name doc-parser-ecs-task-execution \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

# Inline policy: read secrets from Secrets Manager
aws iam put-role-policy \
  --role-name doc-parser-ecs-task-execution \
  --policy-name secrets-manager-read \
  --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [{
      \"Effect\": \"Allow\",
      \"Action\": [\"secretsmanager:GetSecretValue\"],
      \"Resource\": \"arn:aws:secretsmanager:${AWS_REGION}:${AWS_ACCOUNT_ID}:secret:doc-parser/*\"
    }]
  }"

# Inline policy: mount EFS volumes
aws iam put-role-policy \
  --role-name doc-parser-ecs-task-execution \
  --policy-name efs-mount \
  --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [{
      \"Effect\": \"Allow\",
      \"Action\": [
        \"elasticfilesystem:ClientMount\",
        \"elasticfilesystem:ClientWrite\",
        \"elasticfilesystem:DescribeMountTargets\"
      ],
      \"Resource\": \"arn:aws:elasticfilesystem:${AWS_REGION}:${AWS_ACCOUNT_ID}:file-system/${FS_ID}\"
    }]
  }"

# Inline policy: ECS Exec (lets you shell into a running container for debugging)
aws iam put-role-policy \
  --role-name doc-parser-ecs-task-execution \
  --policy-name ecs-exec \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Action": [
        "ssmmessages:CreateControlChannel",
        "ssmmessages:CreateDataChannel",
        "ssmmessages:OpenControlChannel",
        "ssmmessages:OpenDataChannel"
      ],
      "Resource": "*"
    }]
  }'

export EXECUTION_ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/doc-parser-ecs-task-execution"
echo "Execution Role ARN: $EXECUTION_ROLE_ARN"
```

```bash
# ✅ Verify all policies are attached
aws iam list-attached-role-policies \
  --role-name doc-parser-ecs-task-execution \
  --query 'AttachedPolicies[*].PolicyName' --output table
# Expected: AmazonECSTaskExecutionRolePolicy

aws iam list-role-policies \
  --role-name doc-parser-ecs-task-execution \
  --query 'PolicyNames' --output table
# Expected: secrets-manager-read, efs-mount, ecs-exec
```

> If `secrets-manager-read` is missing, ECS tasks will fail at startup with `TaskFailedToStart: AccessDeniedException` on Secrets Manager before any container even launches. Re-run the `put-role-policy` command above and ECS will retry automatically.

---

## 11. CloudWatch Log Groups

```bash
aws logs create-log-group \
  --log-group-name /ecs/doc-parser-app \
  --region $AWS_REGION
```

---

## 12. ECS Task Definitions

### 12.1 App Task Definition

This task runs three containers:
- **app** — FastAPI backend (port 8000)
- **qdrant** — vector database sidecar (port 6333, EFS-backed)
- **ollama** — local LLM / OCR engine (port 11434, EFS-backed, `essential: true`)

> **Key configuration notes:**
> - `memory: 16384` (16 GB) — required to run Ollama + Qdrant + app concurrently
> - `PARSER_BACKEND=ollama` — uses local Ollama for PDF parsing, no Z.AI key needed
> - `QDRANT__STORAGE__SKIP_FILESYNC_ON_OPEN=true` — suppresses Qdrant's NFS warning on EFS (harmless, see Section 18C)
> - `app` depends on qdrant and ollama with condition `HEALTHY` — app only starts after both pass health checks
> - Qdrant healthCheck uses `wget` (not `curl` — curl is not available in the qdrant image)
> - Ollama healthCheck uses `curl` (available in the ollama image)
> - Qdrant image pinned to `v1.17.0` for API compatibility

```bash
cat > /tmp/app-task-def.json << EOF
{
  "family": "doc-parser-app",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "16384",
  "executionRoleArn": "${EXECUTION_ROLE_ARN}",
  "taskRoleArn": "${EXECUTION_ROLE_ARN}",
  "volumes": [
    {
      "name": "qdrant-data",
      "efsVolumeConfiguration": {
        "fileSystemId": "${FS_ID}",
        "transitEncryption": "ENABLED",
        "authorizationConfig": {
          "accessPointId": "${QDRANT_AP}",
          "iam": "ENABLED"
        }
      }
    },
    {
      "name": "ollama-models",
      "efsVolumeConfiguration": {
        "fileSystemId": "${FS_ID}",
        "transitEncryption": "ENABLED",
        "authorizationConfig": {
          "accessPointId": "${OLLAMA_AP}",
          "iam": "ENABLED"
        }
      }
    }
  ],
  "containerDefinitions": [
    {
      "name": "app",
      "image": "${ECR_REGISTRY}/doc-parser/app:latest",
      "portMappings": [{"containerPort": 8000, "protocol": "tcp"}],
      "essential": true,
      "environment": [
        {"name": "EMBEDDING_PROVIDER",  "value": "openai"},
        {"name": "RERANKER_BACKEND",    "value": "openai"},
        {"name": "QDRANT_URL",          "value": "http://localhost:6333"},
        {"name": "PARSER_BACKEND",      "value": "ollama"},
        {"name": "OLLAMA_BASE_URL",     "value": "http://localhost:11434"}
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:${AWS_REGION}:${AWS_ACCOUNT_ID}:secret:doc-parser/openai-api-key:OPENAI_API_KEY::"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group":         "/ecs/doc-parser-app",
          "awslogs-region":        "${AWS_REGION}",
          "awslogs-stream-prefix": "app"
        }
      },
      "healthCheck": {
        "command":     ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval":    30,
        "timeout":     5,
        "retries":     3,
        "startPeriod": 60
      },
      "dependsOn": [
        {"containerName": "qdrant", "condition": "HEALTHY"},
        {"containerName": "ollama", "condition": "HEALTHY"}
      ]
    },
    {
      "name": "qdrant",
      "image": "qdrant/qdrant:v1.17.0",
      "portMappings": [{"containerPort": 6333, "protocol": "tcp"}],
      "essential": true,
      "mountPoints": [
        {
          "sourceVolume":   "qdrant-data",
          "containerPath":  "/qdrant/storage",
          "readOnly":       false
        }
      ],
      "environment": [
        {"name": "QDRANT__STORAGE__SKIP_FILESYNC_ON_OPEN", "value": "true"}
      ],
      "healthCheck": {
        "command":     ["CMD-SHELL", "wget -qO /dev/null http://localhost:6333/healthz || exit 1"],
        "interval":    15,
        "timeout":     5,
        "retries":     3,
        "startPeriod": 30
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group":         "/ecs/doc-parser-app",
          "awslogs-region":        "${AWS_REGION}",
          "awslogs-stream-prefix": "qdrant"
        }
      }
    },
    {
      "name": "ollama",
      "image": "ollama/ollama:latest",
      "portMappings": [{"containerPort": 11434, "protocol": "tcp"}],
      "essential": true,
      "mountPoints": [
        {
          "sourceVolume":  "ollama-models",
          "containerPath": "/root/.ollama",
          "readOnly":      false
        }
      ],
      "environment": [
        {"name": "OLLAMA_HOST", "value": "0.0.0.0"}
      ],
      "healthCheck": {
        "command":     ["CMD-SHELL", "curl -sf http://localhost:11434/ || exit 1"],
        "interval":    15,
        "timeout":     5,
        "retries":     3,
        "startPeriod": 30
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group":         "/ecs/doc-parser-app",
          "awslogs-region":        "${AWS_REGION}",
          "awslogs-stream-prefix": "ollama"
        }
      }
    }
  ]
}
EOF

aws ecs register-task-definition \
  --cli-input-json file:///tmp/app-task-def.json \
  --region $AWS_REGION
```

---

## 13. Application Load Balancer

```bash
# Create the ALB (internet-facing)
ALB_ARN=$(aws elbv2 create-load-balancer \
  --name doc-parser-alb \
  --subnets $(echo $SUBNET_IDS | tr ',' ' ') \
  --security-groups $ALB_SG \
  --scheme internet-facing \
  --type application \
  --query 'LoadBalancers[0].LoadBalancerArn' --output text)
echo "ALB ARN: $ALB_ARN"

# Target group for FastAPI app (port 8000)
APP_TG_ARN=$(aws elbv2 create-target-group \
  --name doc-parser-app-tg \
  --protocol HTTP \
  --port 8000 \
  --target-type ip \
  --vpc-id $VPC_ID \
  --health-check-path /health \
  --query 'TargetGroups[0].TargetGroupArn' --output text)
echo "App TG: $APP_TG_ARN"

# Listener on port 80 — all traffic → FastAPI app
LISTENER_ARN=$(aws elbv2 create-listener \
  --load-balancer-arn $ALB_ARN \
  --protocol HTTP \
  --port 80 \
  --default-actions Type=forward,TargetGroupArn=$APP_TG_ARN \
  --query 'Listeners[0].ListenerArn' --output text)
echo "Listener: $LISTENER_ARN"

# Increase idle timeout to 300s — default 60s causes 504 on /ingest/file
# (PDF parsing + captioning + embedding takes 1-3 minutes)
aws elbv2 modify-load-balancer-attributes \
  --load-balancer-arn $ALB_ARN \
  --attributes Key=idle_timeout.timeout_seconds,Value=300 \
  --region $AWS_REGION

# Print the public URL
ALB_DNS=$(aws elbv2 describe-load-balancers \
  --load-balancer-arns $ALB_ARN \
  --query 'LoadBalancers[0].DNSName' --output text)
echo "Public URL: http://${ALB_DNS}"
```

---

## 14. ECS Services

> **Note:** `assignPublicIp=ENABLED` is required because the default VPC uses public subnets
> without a NAT gateway. Fargate needs outbound internet access to pull images from ECR.

```bash
# App service (FastAPI + Qdrant + Ollama)
aws ecs create-service \
  --cluster $CLUSTER_NAME \
  --service-name doc-parser-app \
  --task-definition doc-parser-app \
  --desired-count 1 \
  --launch-type FARGATE \
  --enable-execute-command \
  --network-configuration "awsvpcConfiguration={
    subnets=[$(echo $SUBNET_IDS | tr ',' ',')],
    securityGroups=[$ECS_SG],
    assignPublicIp=ENABLED
  }" \
  --load-balancers "targetGroupArn=$APP_TG_ARN,containerName=app,containerPort=8000" \
  --region $AWS_REGION
```

Wait for the service to reach a stable state:

```bash
aws ecs wait services-stable \
  --cluster $CLUSTER_NAME \
  --services doc-parser-app
echo "Service is stable."
```

---

## 15. Ollama Model Bootstrap

This step is run **once** after the first deployment. The model is saved to EFS and persists
across all future deployments.

```bash
# Find a running task in the app service
TASK_ARN=$(aws ecs list-tasks \
  --cluster $CLUSTER_NAME \
  --service-name doc-parser-app \
  --query 'taskArns[0]' --output text)
echo "Task: $TASK_ARN"

# Shell into the Ollama container and pull the model
aws ecs execute-command \
  --cluster $CLUSTER_NAME \
  --task $TASK_ARN \
  --container ollama \
  --interactive \
  --command "ollama pull glm4v:9b"
```

> The pull takes 5–10 minutes (the model is ~6 GB). Once complete, it is stored on EFS and
> you will never need to pull it again — even after new deployments.

---

## 16. GitHub Actions Secrets

The CI/CD pipeline needs these secrets set in GitHub:
**Repository → Settings → Secrets and variables → Actions**

```bash
# Use the doc-parser-cicd credentials from Phase 9
gh secret set AWS_ACCESS_KEY_ID     --body "<cicd-access-key-id>"
gh secret set AWS_SECRET_ACCESS_KEY --body "<cicd-secret-access-key>"
gh secret set AWS_REGION            --body "us-east-1"
gh secret set ECR_REGISTRY          --body "${ECR_REGISTRY}"
gh secret set ECS_CLUSTER           --body "doc-parser-cluster"
gh secret set ECS_SERVICE_APP       --body "doc-parser-app"
```

| Secret | Value |
|--------|-------|
| `AWS_ACCESS_KEY_ID` | From Phase 9 (`doc-parser-cicd` access key) |
| `AWS_SECRET_ACCESS_KEY` | From Phase 9 (`doc-parser-cicd` secret key) |
| `AWS_REGION` | `us-east-1` |
| `ECR_REGISTRY` | `685057748560.dkr.ecr.us-east-1.amazonaws.com` |
| `ECS_CLUSTER` | `doc-parser-cluster` |
| `ECS_SERVICE_APP` | `doc-parser-app` |

---

## 17. Verify Deployment

```bash
# Service status
aws ecs describe-services \
  --cluster $CLUSTER_NAME \
  --services doc-parser-app \
  --query 'services[*].{name:serviceName,running:runningCount,desired:desiredCount,status:status}' \
  --output table

# Get public URL
ALB_DNS=$(aws elbv2 describe-load-balancers \
  --names doc-parser-alb \
  --query 'LoadBalancers[0].DNSName' --output text)

# Health check
curl http://${ALB_DNS}/health    # → {"status":"ok"}

# Open in browser
echo "API URL: http://${ALB_DNS}"

# Tail logs
aws logs tail /ecs/doc-parser-app --follow
```

---

## 18. Troubleshooting Common Deployment Issues

### A — Task fails to start: `AccessDeniedException` on Secrets Manager

**Symptom:** Service shows `runningCount: 0` and `failedTasks > 0`. CI/CD loop never stabilises. The task stops before any container launches.

**Diagnose:**
```bash
# Find the stopped task ARN
aws ecs list-tasks --cluster doc-parser-cluster \
  --service-name doc-parser-app --desired-status STOPPED \
  --region us-east-1

# Get the exact failure reason
aws ecs describe-tasks --cluster doc-parser-cluster \
  --tasks <task-arn> --region us-east-1 \
  --query 'tasks[0].{stopCode:stopCode,reason:stoppedReason}'
```

**What you'll see:**
```
stopCode: TaskFailedToStart
reason: ...not authorized to perform: secretsmanager:GetSecretValue on resource: doc-parser/openai-api-key
```

**Fix:** Re-attach the inline policy to the execution role:
```bash
aws iam put-role-policy \
  --role-name doc-parser-ecs-task-execution \
  --policy-name secrets-manager-read \
  --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [{
      \"Effect\": \"Allow\",
      \"Action\": [\"secretsmanager:GetSecretValue\"],
      \"Resource\": \"arn:aws:secretsmanager:${AWS_REGION}:${AWS_ACCOUNT_ID}:secret:doc-parser/*\"
    }]
  }"
```

ECS retries automatically after the policy is attached — no redeployment needed.

---

### B — ALB health checks timing out: `Target.Timeout`

**Symptom:** Task is RUNNING and app logs show uvicorn started on port 8000, but the ALB target stays `unhealthy — Target.Timeout`. No HTTP requests appear in the app logs at all.

**Diagnose:**
```bash
# Step 1 — confirm the target is unhealthy
aws elbv2 describe-target-health \
  --target-group-arn <tg-arn> --region us-east-1 \
  --query 'TargetHealthDescriptions[*].{target:Target.Id,state:TargetHealth.State,reason:TargetHealth.Reason}'

# Step 2 — get the running task's ENI
aws ecs describe-tasks --cluster doc-parser-cluster \
  --tasks <task-id> --region us-east-1 \
  --query 'tasks[0].attachments[0].details'

# Step 3 — find the security group on that ENI
aws ec2 describe-network-interfaces \
  --network-interface-ids <eni-id> --region us-east-1 \
  --query 'NetworkInterfaces[0].Groups'

# Step 4 — check inbound rules (port 8000 from ALB will be missing)
aws ec2 describe-security-groups \
  --group-ids <ecs-sg-id> --region us-east-1 \
  --query 'SecurityGroups[0].IpPermissions'
```

**What you'll see:** port `8000` is absent from the inbound rules.

**Fix:** Add the missing rule:
```bash
ALB_SG=$(aws elbv2 describe-load-balancers --names doc-parser-alb \
  --region us-east-1 --query 'LoadBalancers[0].SecurityGroups[0]' --output text)

aws ec2 authorize-security-group-ingress \
  --group-id <ecs-sg-id> \
  --protocol tcp --port 8000 \
  --source-group $ALB_SG \
  --region us-east-1
```

Takes effect immediately — no redeployment needed. The target flips to healthy within ~2 minutes once the ALB can reach port 8000.

---

### C — Qdrant NFS warning on EFS (not a fatal error)

**Symptom:** Qdrant container logs show:
```
ERROR qdrant: Filesystem check failed for storage path ./storage.
Details: NFS may cause data corruption due to inconsistent file locking
```

This is **expected and harmless**. Qdrant detects that EFS is NFS-backed and warns about file locking. The task definition already sets `QDRANT__STORAGE__SKIP_FILESYNC_ON_OPEN=true` to bypass the fatal check. Qdrant starts normally. No action needed.

---

## 19. CI/CD Flow Reference

```
Push to any branch / PR opened
        │
        ▼
CI workflow (ci.yml)
  ├── ruff check src/ tests/ scripts/
  ├── mypy src/
  └── pytest tests/unit/ -v   (no API keys, ~30s)

Push to main (after PR merge)
        │
        ▼
CD workflow (cd.yml)
  ├── docker build + push app image  → ECR :<git-sha> + :latest
  ├── ecs update-service (app)       → force new deployment
  └── ecs wait services-stable
```

Every deployment registers a new ECS task definition revision, enabling clean rollbacks.

---

## 20. Rollback Procedure

```bash
# List recent task definition revisions
aws ecs list-task-definitions \
  --family-prefix doc-parser-app \
  --sort DESC \
  --query 'taskDefinitionArns[:5]' \
  --output table

# Roll back to a specific revision (replace 7 with the revision number)
aws ecs update-service \
  --cluster $CLUSTER_NAME \
  --service doc-parser-app \
  --task-definition doc-parser-app:7

# Wait for stable
aws ecs wait services-stable \
  --cluster $CLUSTER_NAME \
  --services doc-parser-app
echo "Rollback complete."
```

---

## IAM Principal Summary

| Principal | Type | Used By | Permissions |
|---|---|---|---|
| `doc-parser-admin` | IAM User | You (local CLI) | AdministratorAccess |
| `doc-parser-cicd` | IAM User | GitHub Actions | ECR push + ECS deploy only |
| `doc-parser-ecs-task-execution` | IAM Role | Fargate at runtime | ECR pull, CloudWatch, Secrets, EFS |

---

## 21. Cost Overview — What This Infrastructure Charges Per Month

> **Read this before leaving the infrastructure running overnight or over a weekend.**

The moment the ECS service is running, AWS charges accumulate every hour — even if no one is sending requests. Here is a realistic breakdown for this deployment.

### Fixed charges (running 24 × 7)

| Service | How it charges | ~Monthly cost |
|---------|---------------|--------------|
| **ECS Fargate — 2 vCPU** | $0.04048 per vCPU-hour × 2 × 730 h | ~$59 |
| **ECS Fargate — 16 GB RAM** | $0.004445 per GB-hour × 16 × 730 h | ~$52 |
| **Application Load Balancer** | $0.0225/hour fixed (hourly charge starts the moment the ALB exists) | ~$16 |
| **EFS storage (~10 GB)** | $0.30 per GB-month | ~$3 |
| **CloudWatch Logs** | $0.50 per GB ingested (~2 GB/month) | ~$1 |
| **Secrets Manager** | $0.40 per secret per month | ~$0.40 |
| **ECR storage (~2 GB)** | $0.10 per GB-month | ~$0.20 |
| **Total** | | **~$131/month** |

### Variable charges (usage-dependent)

| Service | Unit cost |
|---------|----------|
| OpenAI embeddings | ~$0.13 / million tokens |
| OpenAI GPT-4o (captioning + answers) | ~$2.50 / million input tokens |
| ALB data processed | ~$0.008 per LCU-hour |
| Data transfer out to internet | $0.09 per GB |

### Key insight for students

**The ALB and Fargate task are the big ticket items.** Together they account for ~$127 of the ~$131/month fixed cost. EFS and the other services are nearly free at this scale. If you forget to stop the infrastructure after a class demo, you will be charged roughly **$4.30 per day** even with zero traffic.

---

## 22. How to Stop the Infrastructure (Save Money, Keep Data)

Use this when you want to **pause** the deployment — stop all charges from Fargate and the ALB, but keep your EFS data (Qdrant vectors + Ollama model) intact so you can resume later without re-ingesting documents or re-downloading the model.

### Step 1 — Scale the ECS service to zero tasks

```bash
aws ecs update-service \
  --cluster doc-parser-cluster \
  --service doc-parser-app \
  --desired-count 0 \
  --region us-east-1

# Confirm it scaled down
aws ecs describe-services \
  --cluster doc-parser-cluster \
  --services doc-parser-app \
  --region us-east-1 \
  --query 'services[0].{running:runningCount,desired:desiredCount}' \
  --output json
# Expected: { "running": 0, "desired": 0 }
```

This stops all Fargate tasks immediately. **Fargate billing stops the moment tasks reach 0.** EFS data is untouched.

### Step 2 — Delete the ALB listener and ALB (stops the $16/month fixed charge)

```bash
# Get the ALB ARN
ALB_ARN=$(aws elbv2 describe-load-balancers \
  --names doc-parser-alb \
  --region us-east-1 \
  --query 'LoadBalancers[0].LoadBalancerArn' \
  --output text)

# Get and delete the listener first
LISTENER_ARN=$(aws elbv2 describe-listeners \
  --load-balancer-arn $ALB_ARN \
  --region us-east-1 \
  --query 'Listeners[0].ListenerArn' \
  --output text)

aws elbv2 delete-listener \
  --listener-arn $LISTENER_ARN \
  --region us-east-1

# Then delete the ALB itself
aws elbv2 delete-load-balancer \
  --load-balancer-arn $ALB_ARN \
  --region us-east-1

echo "ALB deleted. Hourly ALB charge stopped."
```

> **Note:** The target group can remain — it has no cost and recreating it is tedious. Only the ALB itself has an hourly charge.

### What you are paying after stopping

| Service | Status | Monthly cost |
|---------|--------|-------------|
| Fargate | Stopped (0 tasks) | $0 |
| ALB | Deleted | $0 |
| EFS (~10 GB) | Data retained | ~$3 |
| CloudWatch Logs | No new logs | ~$0 |
| Secrets Manager | Secret retained | ~$0.40 |
| ECR | Images retained | ~$0.20 |
| **Total while paused** | | **~$3.60/month** |

---

## 23. How to Restart the Infrastructure

When you want to bring everything back up, follow these steps in order.

### Step 1 — Recreate the ALB

```bash
# Get your subnet IDs and ALB security group ID
SUBNET_1=$(aws ec2 describe-subnets \
  --region us-east-1 \
  --filters "Name=defaultForAz,Values=true" \
  --query 'Subnets[0].SubnetId' --output text)

SUBNET_2=$(aws ec2 describe-subnets \
  --region us-east-1 \
  --filters "Name=defaultForAz,Values=true" \
  --query 'Subnets[1].SubnetId' --output text)

ALB_SG=$(aws ec2 describe-security-groups \
  --region us-east-1 \
  --filters "Name=group-name,Values=doc-parser-alb-sg" \
  --query 'SecurityGroups[0].GroupId' --output text)

# Create the ALB
ALB_ARN=$(aws elbv2 create-load-balancer \
  --name doc-parser-alb \
  --subnets $SUBNET_1 $SUBNET_2 \
  --security-groups $ALB_SG \
  --region us-east-1 \
  --query 'LoadBalancers[0].LoadBalancerArn' \
  --output text)

echo "ALB ARN: $ALB_ARN"
```

### Step 2 — Set the ALB idle timeout to 300 seconds

```bash
aws elbv2 modify-load-balancer-attributes \
  --load-balancer-arn $ALB_ARN \
  --attributes Key=idle_timeout.timeout_seconds,Value=300 \
  --region us-east-1
```

### Step 3 — Recreate the listener pointing to the existing target group

```bash
TG_ARN=$(aws elbv2 describe-target-groups \
  --names doc-parser-app-tg \
  --region us-east-1 \
  --query 'TargetGroups[0].TargetGroupArn' \
  --output text)

aws elbv2 create-listener \
  --load-balancer-arn $ALB_ARN \
  --protocol HTTP \
  --port 80 \
  --default-actions Type=forward,TargetGroupArn=$TG_ARN \
  --region us-east-1

echo "Listener created."
```

### Step 4 — Scale the ECS service back up

```bash
aws ecs update-service \
  --cluster doc-parser-cluster \
  --service doc-parser-app \
  --desired-count 1 \
  --region us-east-1

# Wait for the task to be healthy (takes ~2-3 minutes)
aws ecs wait services-stable \
  --cluster doc-parser-cluster \
  --services doc-parser-app \
  --region us-east-1 && echo "Service is stable and healthy."
```

### Step 5 — Get the new ALB DNS name and verify

```bash
NEW_DNS=$(aws elbv2 describe-load-balancers \
  --names doc-parser-alb \
  --region us-east-1 \
  --query 'LoadBalancers[0].DNSName' \
  --output text)

echo "New ALB URL: http://$NEW_DNS"

# Verify health
curl -s "http://$NEW_DNS/health" | python3 -m json.tool
```

> **Note:** The ALB DNS name changes each time you recreate it. Update any bookmarks or client configurations with the new URL.

---

## 24. How to Tear Down Everything (Full Deletion)

> **Warning — this is irreversible.** All Qdrant vector data, all ingested documents, and all infrastructure will be permanently deleted. You will need to start from Phase 1 of this guide to redeploy.

Run these commands in order. Each step depends on resources from the previous steps being removed first.

### Step 1 — Stop and delete the ECS service

```bash
# Scale to zero first (graceful shutdown)
aws ecs update-service \
  --cluster doc-parser-cluster \
  --service doc-parser-app \
  --desired-count 0 \
  --region us-east-1

# Wait for all tasks to stop
aws ecs wait services-stable \
  --cluster doc-parser-cluster \
  --services doc-parser-app \
  --region us-east-1

# Delete the service
aws ecs delete-service \
  --cluster doc-parser-cluster \
  --service doc-parser-app \
  --region us-east-1

echo "ECS service deleted."
```

### Step 2 — Delete the ALB, listener, and target group

```bash
# Get ALB ARN
ALB_ARN=$(aws elbv2 describe-load-balancers \
  --names doc-parser-alb \
  --region us-east-1 \
  --query 'LoadBalancers[0].LoadBalancerArn' \
  --output text)

# Delete listener
LISTENER_ARN=$(aws elbv2 describe-listeners \
  --load-balancer-arn $ALB_ARN \
  --region us-east-1 \
  --query 'Listeners[0].ListenerArn' \
  --output text)

aws elbv2 delete-listener --listener-arn $LISTENER_ARN --region us-east-1

# Delete ALB
aws elbv2 delete-load-balancer --load-balancer-arn $ALB_ARN --region us-east-1

# Delete target group
TG_ARN=$(aws elbv2 describe-target-groups \
  --names doc-parser-app-tg \
  --region us-east-1 \
  --query 'TargetGroups[0].TargetGroupArn' \
  --output text)

aws elbv2 delete-target-group --target-group-arn $TG_ARN --region us-east-1

echo "ALB, listener, and target group deleted."
```

### Step 3 — Delete the EFS filesystem (permanent data loss)

```bash
# Delete EFS access points first
for AP in $(aws efs describe-access-points \
  --file-system-id fs-037be44eb75b00b25 \
  --region us-east-1 \
  --query 'AccessPoints[*].AccessPointId' \
  --output text); do
  aws efs delete-access-point --access-point-id $AP --region us-east-1
  echo "Deleted access point: $AP"
done

# Delete EFS mount targets
for MT in $(aws efs describe-mount-targets \
  --file-system-id fs-037be44eb75b00b25 \
  --region us-east-1 \
  --query 'MountTargets[*].MountTargetId' \
  --output text); do
  aws efs delete-mount-target --mount-target-id $MT --region us-east-1
  echo "Deleted mount target: $MT"
done

# Wait for mount targets to be fully deleted (takes ~30 seconds)
echo "Waiting 45 seconds for mount targets to finish deleting..."
sleep 45

# Delete the filesystem itself
aws efs delete-file-system \
  --file-system-id fs-037be44eb75b00b25 \
  --region us-east-1

echo "EFS filesystem deleted. All Qdrant and Ollama data is gone."
```

### Step 4 — Delete the ECS cluster

```bash
aws ecs delete-cluster \
  --cluster doc-parser-cluster \
  --region us-east-1

echo "ECS cluster deleted."
```

### Step 5 — Delete ECR repositories and images

```bash
# Delete all images first, then the repository
aws ecr batch-delete-image \
  --repository-name doc-parser/app \
  --region us-east-1 \
  --image-ids "$(aws ecr list-images \
    --repository-name doc-parser/app \
    --region us-east-1 \
    --query 'imageIds' \
    --output json)"

aws ecr delete-repository \
  --repository-name doc-parser/app \
  --region us-east-1 \
  --force

echo "ECR repository deleted."
```

### Step 6 — Delete CloudWatch log group

```bash
aws logs delete-log-group \
  --log-group-name /ecs/doc-parser-app \
  --region us-east-1

echo "CloudWatch log group deleted."
```

### Step 7 — Delete Secrets Manager secret

```bash
# force-delete-without-recovery skips the 30-day recovery window
aws secretsmanager delete-secret \
  --secret-id doc-parser/openai-api-key \
  --force-delete-without-recovery \
  --region us-east-1

echo "Secret deleted."
```

### Step 8 — Delete IAM role and policies

```bash
# Detach managed policies
aws iam detach-role-policy \
  --role-name doc-parser-ecs-task-execution \
  --policy-arn arn:aws:iam::aws:policy/AmazonECSTaskExecutionRolePolicy

aws iam detach-role-policy \
  --role-name doc-parser-ecs-task-execution \
  --policy-arn arn:aws:iam::aws:policy/AmazonElasticFileSystemClientFullAccess

# Delete inline policies
for POLICY in $(aws iam list-role-policies \
  --role-name doc-parser-ecs-task-execution \
  --query 'PolicyNames' \
  --output text); do
  aws iam delete-role-policy \
    --role-name doc-parser-ecs-task-execution \
    --policy-name "$POLICY"
  echo "Deleted inline policy: $POLICY"
done

# Delete the role
aws iam delete-role \
  --role-name doc-parser-ecs-task-execution

echo "IAM role deleted."
```

### Step 9 — Delete Security Groups

```bash
# Get security group IDs
ALB_SG=$(aws ec2 describe-security-groups \
  --region us-east-1 \
  --filters "Name=group-name,Values=doc-parser-alb-sg" \
  --query 'SecurityGroups[0].GroupId' --output text)

ECS_SG=$(aws ec2 describe-security-groups \
  --region us-east-1 \
  --filters "Name=group-name,Values=doc-parser-ecs-sg" \
  --query 'SecurityGroups[0].GroupId' --output text)

# Delete ECS SG first (ALB SG has a dependency on it via inbound rules)
aws ec2 delete-security-group --group-id $ECS_SG --region us-east-1
aws ec2 delete-security-group --group-id $ALB_SG --region us-east-1

echo "Security groups deleted."
```

### Step 10 — Delete IAM users (optional)

Only do this if you want to fully clean up the account. If you plan to redeploy later, keep the users.

```bash
# Remove doc-parser-cicd user
aws iam detach-user-policy \
  --user-name doc-parser-cicd \
  --policy-arn arn:aws:iam::685057748560:policy/doc-parser-cicd-policy 2>/dev/null || true

for KEY in $(aws iam list-access-keys \
  --user-name doc-parser-cicd \
  --query 'AccessKeyMetadata[*].AccessKeyId' \
  --output text); do
  aws iam delete-access-key --user-name doc-parser-cicd --access-key-id $KEY
done

aws iam delete-user --user-name doc-parser-cicd

echo "doc-parser-cicd user deleted."
```

### Verify everything is gone

```bash
echo "=== Remaining ECS services ==="
aws ecs list-services --cluster doc-parser-cluster --region us-east-1 2>&1

echo "=== Remaining ALBs ==="
aws elbv2 describe-load-balancers \
  --names doc-parser-alb --region us-east-1 2>&1 | grep -E "DNSName|does not exist"

echo "=== Remaining EFS ==="
aws efs describe-file-systems \
  --file-system-id fs-037be44eb75b00b25 \
  --region us-east-1 2>&1 | grep -E "LifeCycleState|does not exist"

echo "Done. All resources deleted."
```

### After full teardown — what still costs money?

| Resource | Cost after teardown |
|----------|-------------------|
| CloudWatch Logs (stored data) | $0.03/GB/month — delete log group in Step 6 to avoid this |
| ECR images | $0 after Step 5 |
| Secrets Manager | $0 after Step 7 (recovery window still counts if not force-deleted) |
| EFS | $0 after Step 3 |
| IAM users / roles | Always free |
| **Total after full teardown** | **$0** |

---

## Quick Reference — Stop vs Pause vs Delete

| Goal | Action | Ongoing cost | Data preserved? | Time to restore |
|------|--------|-------------|----------------|----------------|
| **Save max money overnight** | Phase 21 (scale to 0 + delete ALB) | ~$3.60/month | Yes | ~5 min |
| **Pause for a few days** | Phase 21 (scale to 0 only) | ~$19/month | Yes | ~3 min |
| **Full teardown** | Phase 23 (all steps) | $0 | No | ~2 hours (full redeploy) |
| **Keep running** | Do nothing | ~$131/month | Yes | N/A |

*Last updated: 2026-04-14 | Stack: ECS Fargate + ECR + EFS + ALB + Secrets Manager | Parser: Ollama (glm4v:9b)*
