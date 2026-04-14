# Debugging Runbook — doc-parser on AWS ECS Fargate

All commands assume:
- **Account ID**: `685057748560`
- **Region**: `us-east-1`
- **Cluster**: `doc-parser-cluster`
- **Service**: `doc-parser-app`
- **ALB DNS**: `doc-parser-alb-2100201665.us-east-1.elb.amazonaws.com`
- **Log group**: `/ecs/doc-parser-app`

---

## 1. Quick Health Check

```bash
# Is the API up?
curl -s http://doc-parser-alb-2100201665.us-east-1.elb.amazonaws.com/health | python3 -m json.tool

# Expected response:
# { "status": "ok", "qdrant": "ok", "openai": "ok", "reranker_backend": "openai" }
```

---

## 2. ECS Service Status

```bash
# Running / desired / pending counts + last 5 events
aws ecs describe-services \
  --cluster doc-parser-cluster \
  --services doc-parser-app \
  --region us-east-1 \
  --query 'services[0].{status:status,running:runningCount,desired:desiredCount,pending:pendingCount,taskDef:taskDefinition,events:events[:5]}' \
  --output json
```

```bash
# Which task revision is currently deployed?
aws ecs describe-services \
  --cluster doc-parser-cluster \
  --services doc-parser-app \
  --region us-east-1 \
  --query 'services[0].taskDefinition' \
  --output text
```

```bash
# Get the running task ARN
aws ecs list-tasks \
  --cluster doc-parser-cluster \
  --service-name doc-parser-app \
  --desired-status RUNNING \
  --region us-east-1 \
  --query 'taskArns' \
  --output text
```

---

## 3. Container Logs

### 3a. Find the latest log stream for each container

```bash
# App container streams (most recently created first)
aws logs describe-log-streams \
  --log-group-name /ecs/doc-parser-app \
  --log-stream-name-prefix "app/app/" \
  --region us-east-1 \
  --query 'logStreams[-1].logStreamName' \
  --output text

# Qdrant streams
aws logs describe-log-streams \
  --log-group-name /ecs/doc-parser-app \
  --log-stream-name-prefix "qdrant/qdrant/" \
  --region us-east-1 \
  --query 'logStreams[-1].logStreamName' \
  --output text

# Ollama streams
aws logs describe-log-streams \
  --log-group-name /ecs/doc-parser-app \
  --log-stream-name-prefix "ollama/ollama/" \
  --region us-east-1 \
  --query 'logStreams[-1].logStreamName' \
  --output text
```

### 3b. Tail the latest app logs (excluding health checks)

```bash
STREAM=$(aws logs describe-log-streams \
  --log-group-name /ecs/doc-parser-app \
  --log-stream-name-prefix "app/app/" \
  --region us-east-1 \
  --query 'logStreams[-1].logStreamName' \
  --output text)

aws logs get-log-events \
  --log-group-name /ecs/doc-parser-app \
  --log-stream-name "$STREAM" \
  --region us-east-1 \
  --limit 100 \
  --query 'events[*].message' \
  --output text | sed 's/\x1b\[[0-9;]*m//g' | tr '\t' '\n' | grep -v "GET /health" | tail -50
```

### 3c. Tail the latest Qdrant logs

```bash
STREAM=$(aws logs describe-log-streams \
  --log-group-name /ecs/doc-parser-app \
  --log-stream-name-prefix "qdrant/qdrant/" \
  --region us-east-1 \
  --query 'logStreams[-1].logStreamName' \
  --output text)

aws logs get-log-events \
  --log-group-name /ecs/doc-parser-app \
  --log-stream-name "$STREAM" \
  --region us-east-1 \
  --limit 50 \
  --query 'events[*].message' \
  --output text | tr '\t' '\n' | tail -30
```

### 3d. Tail the latest Ollama logs

```bash
STREAM=$(aws logs describe-log-streams \
  --log-group-name /ecs/doc-parser-app \
  --log-stream-name-prefix "ollama/ollama/" \
  --region us-east-1 \
  --query 'logStreams[-1].logStreamName' \
  --output text)

aws logs get-log-events \
  --log-group-name /ecs/doc-parser-app \
  --log-stream-name "$STREAM" \
  --region us-east-1 \
  --limit 50 \
  --query 'events[*].message' \
  --output text | tr '\t' '\n' | tail -30
```

---

## 4. Why Did the Task Die? (Stopped Tasks)

```bash
# List recently stopped tasks
aws ecs list-tasks \
  --cluster doc-parser-cluster \
  --desired-status STOPPED \
  --region us-east-1 \
  --query 'taskArns[:5]' \
  --output json
```

```bash
# Get stop reason + exit codes for a specific task
TASK_ARN="arn:aws:ecs:us-east-1:685057748560:task/doc-parser-cluster/<TASK_ID>"

aws ecs describe-tasks \
  --cluster doc-parser-cluster \
  --tasks "$TASK_ARN" \
  --region us-east-1 \
  --query 'tasks[0].{stopCode:stopCode,reason:stoppedReason,containers:containers[*].{name:name,exit:exitCode,reason:reason}}' \
  --output json
```

**Exit code reference:**

| Exit code | Meaning |
|-----------|---------|
| `137` | OOM kill — container exceeded memory limit |
| `1` | Process crashed (check logs for Python traceback) |
| `0` | Clean exit (normal shutdown) |
| `143` | SIGTERM — graceful stop (usually fine) |
| `255` | `exec format error` — wrong CPU arch (need `--platform linux/amd64` on Apple Silicon) |

---

## 5. Shell Access into Running Containers (ECS Exec)

> **Prerequisite**: `brew install --cask session-manager-plugin` must be installed on your Mac.

```bash
# Get the running task ID first
TASK_ID=$(aws ecs list-tasks \
  --cluster doc-parser-cluster \
  --service-name doc-parser-app \
  --desired-status RUNNING \
  --region us-east-1 \
  --query 'taskArns[0]' \
  --output text | awk -F/ '{print $NF}')

echo "Task ID: $TASK_ID"
```

```bash
# Shell into the app container
aws ecs execute-command \
  --cluster doc-parser-cluster \
  --task "$TASK_ID" \
  --container app \
  --interactive \
  --command "/bin/bash" \
  --region us-east-1
```

```bash
# Shell into the Qdrant container
aws ecs execute-command \
  --cluster doc-parser-cluster \
  --task "$TASK_ID" \
  --container qdrant \
  --interactive \
  --command "/bin/bash" \
  --region us-east-1
```

```bash
# Shell into the Ollama container
aws ecs execute-command \
  --cluster doc-parser-cluster \
  --task "$TASK_ID" \
  --container ollama \
  --interactive \
  --command "/bin/bash" \
  --region us-east-1
```

---

## 6. Qdrant Dashboard

Qdrant's web UI runs on port 6333 at the `/dashboard` path. Since it is not exposed via the ALB, you need an SSH tunnel via the running ECS task.

### Method: AWS SSM port forward

```bash
# Get the running task ID
TASK_ID=$(aws ecs list-tasks \
  --cluster doc-parser-cluster \
  --service-name doc-parser-app \
  --desired-status RUNNING \
  --region us-east-1 \
  --query 'taskArns[0]' \
  --output text | awk -F/ '{print $NF}')

# Forward localhost:6333 → Qdrant container port 6333 inside the task
aws ssm start-session \
  --target "ecs:doc-parser-cluster_${TASK_ID}_$(aws ecs describe-tasks \
    --cluster doc-parser-cluster \
    --tasks "$TASK_ID" \
    --region us-east-1 \
    --query 'tasks[0].containers[?name==`qdrant`].runtimeId' \
    --output text)" \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["6333"],"localPortNumber":["6333"]}' \
  --region us-east-1
```

Then open your browser at: **http://localhost:6333/dashboard**

> The tunnel stays open as long as the command runs. Press `Ctrl+C` to close it.

### Qdrant REST API (via ALB — not exposed, use curl inside ECS Exec instead)

```bash
# From inside the app or qdrant container shell:
curl http://localhost:6333/collections
curl http://localhost:6333/collections/documents
curl http://localhost:6333/healthz
```

---

## 7. Ollama Management

### Check which models are downloaded

```bash
# Via ECS Exec shell into the ollama container:
ollama list
```

### Pull a model (if missing from EFS)

```bash
# Inside ollama container shell:
ollama pull glm-ocr:latest
```

### Check Ollama API from inside the app container

```bash
# Inside app container shell:
curl http://localhost:11434/api/tags        # list models
curl http://localhost:11434/api/version     # Ollama version
```

---

## 8. ALB / Target Group Health

```bash
# Get the target group ARN
TG_ARN=$(aws elbv2 describe-target-groups \
  --names doc-parser-app-tg \
  --region us-east-1 \
  --query 'TargetGroups[0].TargetGroupArn' \
  --output text)

# Check target health
aws elbv2 describe-target-health \
  --target-group-arn "$TG_ARN" \
  --region us-east-1 \
  --output json
```

**Target states:**

| State | Meaning |
|-------|---------|
| `healthy` | ALB is routing traffic to this target |
| `initial` | Target just registered, health checks haven't passed yet |
| `unhealthy` | Health check failing — check app logs |
| `draining` | Task is being replaced, connections draining |

```bash
# Check ALB idle timeout (should be 300s)
ALB_ARN=$(aws elbv2 describe-load-balancers \
  --names doc-parser-alb \
  --region us-east-1 \
  --query 'LoadBalancers[0].LoadBalancerArn' \
  --output text)

aws elbv2 describe-load-balancer-attributes \
  --load-balancer-arn "$ALB_ARN" \
  --region us-east-1 \
  --query 'Attributes[?Key==`idle_timeout.timeout_seconds`]' \
  --output json
```

---

## 9. Force a New Deployment

```bash
# Force ECS to replace the running task (useful when container is stuck)
aws ecs update-service \
  --cluster doc-parser-cluster \
  --service doc-parser-app \
  --force-new-deployment \
  --region us-east-1 \
  --query 'service.{status:status,desiredCount:desiredCount}' \
  --output json
```

```bash
# Wait until stable
aws ecs wait services-stable \
  --cluster doc-parser-cluster \
  --services doc-parser-app \
  --region us-east-1 && echo "Service is stable"
```

---

## 10. Redeploy After a Docker Image Update

```bash
# 1. Build for linux/amd64 (required on Apple Silicon Macs)
docker build --platform linux/amd64 -t doc-parser/app:latest .

# 2. Authenticate to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  685057748560.dkr.ecr.us-east-1.amazonaws.com

# 3. Tag and push
docker tag doc-parser/app:latest \
  685057748560.dkr.ecr.us-east-1.amazonaws.com/doc-parser/app:latest

docker push 685057748560.dkr.ecr.us-east-1.amazonaws.com/doc-parser/app:latest

# 4. Force new deployment (ECS pulls the new :latest image)
aws ecs update-service \
  --cluster doc-parser-cluster \
  --service doc-parser-app \
  --force-new-deployment \
  --region us-east-1 > /dev/null

aws ecs wait services-stable \
  --cluster doc-parser-cluster \
  --services doc-parser-app \
  --region us-east-1 && echo "Deployed"
```

---

## 11. Update Task Definition (CPU / Memory / Env Vars)

```bash
# 1. Edit app-task-def.json (in project root), then register:
aws ecs register-task-definition \
  --cli-input-json file://app-task-def.json \
  --region us-east-1 \
  --query 'taskDefinition.{family:family,revision:revision,cpu:cpu,memory:memory}' \
  --output json

# 2. Update the service to use the new revision
aws ecs update-service \
  --cluster doc-parser-cluster \
  --service doc-parser-app \
  --task-definition doc-parser-app:<REVISION> \
  --region us-east-1 > /dev/null

aws ecs wait services-stable \
  --cluster doc-parser-cluster \
  --services doc-parser-app \
  --region us-east-1 && echo "Stable on new revision"
```

**Valid Fargate CPU / Memory combinations:**

| CPU (vCPU) | Memory options |
|------------|---------------|
| 512 (0.25) | 1–2 GB |
| 1024 (0.5) | 2–4 GB |
| 2048 (1) | 4–16 GB |
| 4096 (2) | 8–30 GB |
| 8192 (4) | 16–60 GB |

Current config: **2048 CPU / 16384 MB (16 GB)**

---

## 12. Common Errors and Fixes

| Symptom | Root cause | Fix |
|---------|-----------|-----|
| `502 Bad Gateway` | App crashed mid-request (OOM or exception) | Check stopped task exit code (`exit 137` = OOM → increase memory) |
| `504 Gateway Timeout` | Request took longer than ALB idle timeout | Set ALB `idle_timeout.timeout_seconds` to 300 |
| `503 Service Unavailable` | No healthy targets registered | Wait for task to pass health checks; check app logs |
| Task stuck in `PENDING` | Dependency (`qdrant`, `ollama`) didn't reach `START` condition | Check Qdrant/Ollama logs for startup errors |
| `exec format error` (exit 255) | Docker image built for ARM64 on Apple Silicon | Rebuild with `--platform linux/amd64` |
| `AccessDeniedException` (Secrets Manager) | Task execution role policy has wrong account ID | Fix IAM inline policy ARNs |
| `Role is not valid` (register task def) | Account ID typo in ARNs inside task definition JSON | Check `executionRoleArn` / `taskRoleArn` in `app-task-def.json` |
| Qdrant health check fails | Qdrant image has no `curl` | Remove health check from Qdrant container; use `dependsOn: START` not `HEALTHY` |
| `pull model manifest: file does not exist` | Wrong Ollama model name | Correct name is `glm-ocr:latest` (not `glm4v:9b`) |
| `AWSServiceRoleForECS does not exist` | First-time ECS use in account | `aws iam create-service-linked-role --aws-service-name ecs.amazonaws.com` |

---

## 13. Secrets Manager

```bash
# Verify the OpenAI API key secret exists
aws secretsmanager describe-secret \
  --secret-id doc-parser/openai-api-key \
  --region us-east-1 \
  --query '{name:Name,arn:ARN}' \
  --output json

# Rotate / update the API key value
aws secretsmanager put-secret-value \
  --secret-id doc-parser/openai-api-key \
  --secret-string '{"OPENAI_API_KEY":"sk-...new-key..."}' \
  --region us-east-1
# Then force-new-deployment so the new task picks up the updated secret.
```

---

## 14. Test the Full Pipeline via curl

```bash
BASE="http://doc-parser-alb-2100201665.us-east-1.elb.amazonaws.com"

# Health
curl -s "$BASE/health" | python3 -m json.tool

# List collections
curl -s "$BASE/collections" | python3 -m json.tool

# Ingest a PDF
curl -X POST "$BASE/ingest/file" \
  -H 'accept: application/json' \
  -F 'file=@/path/to/your.pdf;type=application/pdf' \
  -F 'overwrite=false' \
  -F 'max_chunk_tokens=512' \
  -F 'caption=true'

# Search (retrieval only)
curl -X POST "$BASE/search" \
  -H 'Content-Type: application/json' \
  -d '{"query": "your question", "top_k": 5, "top_n": 3, "rerank": true}'

# Generate (RAG answer)
curl -X POST "$BASE/generate" \
  -H 'Content-Type: application/json' \
  -d '{"query": "your question", "top_k": 5, "top_n": 3, "rerank": true, "max_tokens": 1024}'

# Delete a collection (irreversible)
curl -X DELETE "$BASE/collections/documents"
```
