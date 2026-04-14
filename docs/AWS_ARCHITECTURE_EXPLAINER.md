# AWS Architecture Explainer — Multi-Modal RAG Pipeline
### A Solution Architect's Guide for Developers and Students

---

## Table of Contents

1. [What We Built — The Big Picture](#1-what-we-built--the-big-picture)
2. [The Full Architecture Diagram (Text)](#2-the-full-architecture-diagram-text)
3. [Every AWS Service Used — and Why](#3-every-aws-service-used--and-why)
4. [Networking Deep Dive — VPC, Subnets, Security Groups](#4-networking-deep-dive--vpc-subnets-security-groups)
5. [The Multi-Container Sidecar Pattern](#5-the-multi-container-sidecar-pattern)
6. [Storage Strategy — Why EFS Over EBS or S3](#6-storage-strategy--why-efs-over-ebs-or-s3)
7. [Security Design](#7-security-design)
8. [Is This Enterprise-Grade?](#8-is-this-enterprise-grade)
9. [Rough Monthly Cost Estimate](#9-rough-monthly-cost-estimate)
10. [Future Enhancements — The Production Roadmap](#10-future-enhancements--the-production-roadmap)
11. [Key Architectural Trade-offs Made](#11-key-architectural-trade-offs-made)
12. [Glossary for Students](#12-glossary-for-students)

---

## 1. What We Built — The Big Picture

We deployed a **Multi-Modal Retrieval-Augmented Generation (RAG) pipeline** that can:

- Accept PDF and image file uploads via a REST API
- Parse documents using a local OCR model (GLM-OCR running inside Ollama)
- Detect document layout (tables, figures, headings) using PP-DocLayoutV3
- Generate captions for figures using GPT-4o (OpenAI)
- Embed text using OpenAI's `text-embedding-3-large` model
- Store vectors in Qdrant (a high-performance vector database)
- Answer natural language questions by retrieving relevant chunks and generating answers with GPT-4o

The entire system runs as a **single Fargate task** with three co-located containers on AWS, reachable through a public URL via an Application Load Balancer.

---

## 2. The Full Architecture Diagram (Text)

```
                         ┌─────────────────────────────────────────────────────┐
                         │                    INTERNET                          │
                         └──────────────────────┬──────────────────────────────┘
                                                │  HTTP :80
                         ┌──────────────────────▼──────────────────────────────┐
                         │         Application Load Balancer (ALB)             │
                         │    doc-parser-alb-2100201665.us-east-1.elb.          │
                         │    amazonaws.com                                     │
                         │    Listener: port 80 → forward to target group       │
                         │    Idle timeout: 300 seconds                         │
                         └──────────────────────┬──────────────────────────────┘
                                                │
                         ┌──────────────────────▼──────────────────────────────┐
                         │           Target Group (doc-parser-app-tg)          │
                         │    Health check: GET /health every 30s               │
                         │    Protocol: HTTP, Port: 8000                        │
                         └──────────────────────┬──────────────────────────────┘
                                                │
                    ┌───────────────────────────▼────────────────────────────────┐
                    │                AWS Default VPC (172.31.0.0/16)             │
                    │                                                             │
                    │   ┌─────────────────────────────────────────────────────┐  │
                    │   │         ECS Fargate Task (2 vCPU / 16 GB RAM)       │  │
                    │   │                                                      │  │
                    │   │  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │  │
                    │   │  │  app:8000    │  │ qdrant:6333  │  │ollama:    │  │  │
                    │   │  │  FastAPI     │  │ Vector DB    │  │11434      │  │  │
                    │   │  │  Python 3.12 │  │ v1.17.0      │  │GLM-OCR    │  │  │
                    │   │  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘  │  │
                    │   │         │                  │                │         │  │
                    │   │         └──────── localhost ───────────────┘         │  │
                    │   │                     (awsvpc)                         │  │
                    │   └──────────────────────┬──────────────────────────────┘  │
                    │                          │                                  │
                    │      ┌───────────────────┴──────────────────┐              │
                    │      │              EFS (NFS)                │              │
                    │      │  /qdrant-data  (access point 1)       │              │
                    │      │  /ollama-models (access point 2)      │              │
                    │      └──────────────────────────────────────┘              │
                    └────────────────────────────────────────────────────────────┘
                                                │
                    ┌───────────────────────────▼────────────────────────────────┐
                    │                     AWS Services                           │
                    │                                                             │
                    │  ECR (container images)    Secrets Manager (OpenAI key)    │
                    │  CloudWatch Logs           IAM Roles                       │
                    └────────────────────────────────────────────────────────────┘
                                                │
                    ┌───────────────────────────▼────────────────────────────────┐
                    │                     EXTERNAL APIs                          │
                    │     api.openai.com (embeddings, captioning, reranking)      │
                    │     huggingface.co (PP-DocLayoutV3 model weights)           │
                    └────────────────────────────────────────────────────────────┘
```

---

## 3. Every AWS Service Used — and Why

### 3.1 Amazon ECS Fargate

**What it is:** A serverless container execution engine. You describe *what* to run (a Docker image, CPU, RAM) and AWS runs it — no virtual machines to provision, patch, or manage.

**Why we chose it over alternatives:**

| Option | Why not? |
|--------|----------|
| EC2 directly | You manage OS patches, Docker installation, instance health. More ops burden. |
| EC2 + ECS (capacity provider) | Still need to manage the EC2 fleet size. |
| AWS Lambda | 15-minute max timeout. Our ingest pipeline takes 2-5 minutes. Also 10 GB RAM max, no persistent filesystem. |
| Kubernetes (EKS) | Overkill for a single service. $72/month just for the control plane, plus node management. |
| **Fargate** ✓ | No servers. Pay only for what runs. Handles task placement, networking, health replacement automatically. |

**How it works:** You register a *task definition* (a JSON file describing your containers, their images, CPU, RAM, volumes, environment variables, logging). ECS reads this and places the task on Fargate's managed compute. If the task dies, ECS automatically starts a replacement.

**Key concept — task vs service:**
- A **task** is one running instance of your containers.
- A **service** wraps a task definition and ensures a desired number of tasks are always running. If a task fails its health check, the service starts a replacement before stopping the old one (rolling replacement, zero downtime).

---

### 3.2 Application Load Balancer (ALB)

**What it is:** A managed Layer 7 (HTTP/HTTPS) load balancer that sits in front of your application and routes incoming requests to healthy backend targets.

**Why we need it even with a single task:**

1. **Health-aware routing** — ALB continuously health-checks your container (`GET /health`). If it fails, ALB stops sending traffic before ECS even replaces the task. Users get zero broken requests.
2. **Stable public URL** — Fargate tasks get ephemeral private IPs that change every deployment. The ALB gives you a stable DNS name that never changes.
3. **SSL termination** — When you add HTTPS later, the ALB handles the TLS handshake and certificate management. Your app only ever sees plain HTTP internally.
4. **Path-based routing** — Future: you could route `/api/*` to one service and `/admin/*` to another, all behind one DNS name.
5. **Idle timeout** — We set this to 300 seconds (5 minutes) because the ingest pipeline takes 2-4 minutes. The default is 60 seconds, which caused 504 timeouts before the fix.

---

### 3.3 Amazon EFS (Elastic File System)

**What it is:** A managed NFS (Network File System) that multiple containers/tasks can mount simultaneously. Data persists indefinitely — it is not tied to any container or task lifecycle.

**Why we need it:**

Our pipeline has two stateful components that must survive task replacements:
1. **Qdrant** stores all your vector embeddings in `/qdrant/storage`. If we used local container storage, every deployment would wipe your entire indexed knowledge base and you'd have to re-ingest everything.
2. **Ollama** downloads `glm-ocr:latest` (2.2 GB) on first startup. Without EFS, every task restart would re-download 2.2 GB from the internet, adding 5-10 minutes of cold-start time.

EFS solves both problems: data persists across task replacements, and the Ollama model only needs to be downloaded once ever.

**EFS Access Points:** We created two separate access points — one for Qdrant data and one for Ollama models. Access points are like subdirectory mount points with enforced POSIX permissions. This gives each container its own isolated directory on the same filesystem, without containers being able to read each other's data.

---

### 3.4 Amazon ECR (Elastic Container Registry)

**What it is:** A private Docker registry managed by AWS, analogous to Docker Hub but inside your AWS account.

**Why private registry over Docker Hub:**
- Images are stored in the same AWS region as ECS — fast pulls over AWS's internal network, no egress charges.
- IAM-controlled access — only your ECS task execution role can pull images.
- Fargate requires images to be in ECR or a registry accessible from within the VPC. Public Docker Hub works, but ECR is faster and more secure.

---

### 3.5 AWS Secrets Manager

**What it is:** A vault for secrets (API keys, database passwords, tokens) that injects them as environment variables into containers at task startup.

**Why not just put `OPENAI_API_KEY` in the task definition or a `.env` file:**

| Approach | Problem |
|----------|---------|
| Hardcoded in task definition JSON | Task definition is stored in ECS and visible to anyone with IAM read access. Secrets in plaintext. |
| Committed to git | Catastrophic — keys get leaked in git history. |
| Hardcoded in Docker image | Anyone who pulls the ECR image can extract the key. |
| **Secrets Manager** ✓ | The key never appears in any file or git history. ECS fetches it at runtime using the task execution role. The app container receives it as a normal environment variable. Even if someone reads the task definition JSON, they only see the Secrets Manager ARN, not the value. |

---

### 3.6 AWS CloudWatch Logs

**What it is:** Centralized log aggregation for all your containers. Every line written to stdout/stderr by your containers is captured, stored, and queryable.

**Why it matters:**
- Containers are ephemeral — when a task is replaced, local container logs are gone forever. CloudWatch preserves them.
- Multiple tasks (during rolling deployments) write to separate log streams, making it easy to compare behavior across versions.
- You can set retention policies (e.g., delete logs after 30 days) to control costs.
- Foundation for alerting: CloudWatch Alarms can page you when error rates spike.

---

### 3.7 IAM Roles

We use **one role** (`doc-parser-ecs-task-execution`) for both the *execution role* and the *task role*. In production these should be separate:

| Role | Purpose |
|------|---------|
| **Execution role** | Used by ECS itself to start your task: pull Docker images from ECR, fetch secrets from Secrets Manager, write logs to CloudWatch. |
| **Task role** | Used by code *inside* your containers at runtime: if your app calls S3, DynamoDB, or other AWS services, those calls are signed with this role. |

This is a form of **least-privilege** security — the principle that every component should have only the permissions it actually needs, nothing more.

---

## 4. Networking Deep Dive — VPC, Subnets, Security Groups

### 4.1 What is a VPC?

A **Virtual Private Cloud (VPC)** is a logically isolated section of the AWS cloud where you launch resources. Think of it as your own private data center network inside AWS. AWS gives every account a **default VPC** in each region, pre-configured with public subnets in every availability zone.

### 4.2 What We Used (Default VPC)

```
Default VPC: 172.31.0.0/16
  ├── us-east-1a: 172.31.0.0/20   (public subnet)
  ├── us-east-1b: 172.31.16.0/20  (public subnet)
  └── us-east-1c: 172.31.32.0/20  (public subnet)
```

Our Fargate task runs in a public subnet with `assignPublicIp: ENABLED`. This means the task gets a public IP and can:
- Receive traffic from the ALB (same VPC)
- Make outbound calls to OpenAI API, HuggingFace, etc.

### 4.3 What a Custom VPC Would Look Like (Enterprise)

```
Custom VPC: 10.0.0.0/16
  ├── Public subnets (ALB lives here)
  │     ├── us-east-1a: 10.0.1.0/24
  │     └── us-east-1b: 10.0.2.0/24
  │
  └── Private subnets (ECS tasks live here — no public IP)
        ├── us-east-1a: 10.0.11.0/24
        └── us-east-1b: 10.0.12.0/24
              │
              └── NAT Gateway → outbound-only internet access
                  (for OpenAI API calls, HuggingFace downloads)
```

**Why private subnets matter for security:**
- In our current setup, the Fargate task has a public IP. A determined attacker who knew it could try to reach port 8000 directly, bypassing the ALB. Security groups block this, but the exposure exists.
- With private subnets, the task has no public IP at all. It is physically unreachable from the internet. Traffic can only come in via the ALB (which is in the public subnet) and is forwarded through the VPC's internal network.
- Outbound internet calls (OpenAI, HuggingFace) go through a **NAT Gateway** — a managed outbound-only internet gateway in the public subnet.

### 4.4 Security Groups — The Virtual Firewall

Security groups are stateful firewalls attached to resources. We use two:

**ALB security group** — controls what traffic reaches the load balancer:
```
Inbound:  TCP 80  from 0.0.0.0/0  (anyone on the internet)
Outbound: All traffic (to forward to ECS tasks)
```

**ECS task security group** — controls what traffic reaches the containers:
```
Inbound:  TCP 8000 from ALB security group only
          TCP 2049 (NFS) from same security group (for EFS)
Outbound: All traffic (for OpenAI, HuggingFace, etc.)
```

The key insight: the ECS task only accepts port 8000 traffic from the ALB, not from the internet directly. This is enforced at the network level, independent of application code.

### 4.5 awsvpc Network Mode

Our task definition uses `networkMode: awsvpc`. This gives **each Fargate task its own Elastic Network Interface (ENI)** — its own private IP address inside the VPC.

Consequence: all three containers in our task (app, qdrant, ollama) share the same network namespace. They talk to each other over `localhost`, exactly like processes on the same machine. Qdrant is at `http://localhost:6333` and Ollama is at `http://localhost:11434` from the app's perspective.

---

## 5. The Multi-Container Sidecar Pattern

### What is it?

Instead of building one monolithic container with all three services, we run three separate containers **in the same task**. Qdrant and Ollama are *sidecars* — supporting services that handle infrastructure concerns (storage, inference) so the main app container stays clean and focused.

```
┌─────────────────────────────────────────────────────┐
│                ECS Fargate Task                      │
│                                                      │
│  ┌────────────┐   HTTP localhost   ┌──────────────┐  │
│  │    app     │ ──────────────────▶│   qdrant     │  │
│  │ (FastAPI)  │   :6333            │ (vector DB)  │  │
│  │            │ ──────────────────▶│              │  │
│  │            │   HTTP localhost   └──────────────┘  │
│  │            │   :11434           ┌──────────────┐  │
│  │            │ ──────────────────▶│    ollama    │  │
│  │            │                   │  (GLM-OCR)   │  │
│  └────────────┘                   └──────────────┘  │
└─────────────────────────────────────────────────────┘
```

### Why co-locate instead of separate services?

| Concern | Separate Services | Sidecar (same task) |
|---------|------------------|---------------------|
| Latency | Network hop between services (1-5ms per call) | `localhost` — effectively 0ms |
| Cost | Three separate Fargate tasks = 3× base cost | One task, shared CPU/RAM |
| Complexity | Service discovery, load balancers, IAM between services | None — just localhost |
| Suitable for | High-scale production (each service scales independently) | Development, low-to-medium traffic |

For an inference pipeline where the app calls Qdrant and Ollama dozens of times per request, saving 1-5ms per call adds up. At 50 Qdrant calls per ingest, that's 50-250ms saved.

### The `dependsOn` ordering

```json
"dependsOn": [
  {"containerName": "qdrant", "condition": "START"},
  {"containerName": "ollama", "condition": "START"}
]
```

This tells ECS: don't start the `app` container until both `qdrant` and `ollama` have at least started their process. Without this, the app might start and try to connect to Qdrant before Qdrant's port is open, causing a startup crash.

We use `START` (not `HEALTHY`) because Qdrant doesn't have `curl` installed, so we can't run a health check command inside the container. `START` only requires the container process to be running, not necessarily serving traffic yet. The app's own startup logic handles connection retries.

---

## 6. Storage Strategy — Why EFS Over EBS or S3

Our pipeline needs persistent storage for two things: Qdrant's vector index and Ollama's model weights. Here is why EFS is the right choice:

| Storage type | Qdrant data | Ollama models | Notes |
|-------------|-------------|--------------|-------|
| **Container local storage** | ❌ Lost on task replacement | ❌ Re-downloads on restart | Ephemeral. Only for temp files. |
| **EBS (block storage)** | ✓ Persistent | ✓ Persistent | Can only mount to **one task at a time**. Blocks rolling deployments — new task can't start until old task releases the volume. |
| **S3 (object storage)** | ❌ Not mountable as filesystem | ❌ Not mountable | Great for archival, but Qdrant needs a real filesystem. Requires a sync agent. |
| **EFS (network filesystem)** ✓ | ✓ Persistent | ✓ Persistent | Can be mounted by **multiple tasks simultaneously**. Rolling deployments work. Multi-AZ by default. |

EFS works like a shared network drive. Multiple containers can read and write simultaneously. This is critical for zero-downtime deployments: the new task mounts the same EFS volume while the old task is still running, so data is always available.

**EFS cost model:** You pay only for the storage you use ($0.30/GB-month standard tier). No pre-provisioning needed. The Ollama model (2.2 GB) + Qdrant index (variable, starts at ~10 MB for small collections) costs roughly $1-3/month.

---

## 7. Security Design

### What's in place now

1. **Secrets never in code or files** — OpenAI API key lives exclusively in Secrets Manager. The ECS task execution role fetches it at startup and injects it as an environment variable.

2. **ECR is private** — Only the task execution role can pull images. No public Docker Hub with unrestricted access.

3. **Security group layering** — The app container only receives traffic from the ALB, not from arbitrary internet addresses.

4. **IAM least privilege** — The task role only has the permissions it needs: read from Secrets Manager, write to CloudWatch Logs, read/write to EFS.

5. **EFS transit encryption** — Data flowing between the Fargate task and EFS is encrypted in transit (TLS). The access points use IAM authorization.

### What's missing for production

1. **HTTPS/TLS** — Currently HTTP only. In production, add an ACM certificate and an HTTPS listener on port 443. All HTTP traffic on port 80 should redirect to 443.

2. **Authentication on the API** — Currently anyone with the ALB URL can call `/ingest/file` or `/generate`. Add API key authentication (a simple header check) or integrate with AWS Cognito/API Gateway for proper auth.

3. **WAF (Web Application Firewall)** — Attach AWS WAF to the ALB to block common attacks (SQL injection, XSS, rate limiting, IP blocking).

4. **Private subnets** — Move ECS tasks to private subnets so they have no public IP. Add a NAT Gateway for outbound internet access.

5. **VPC Endpoints** — Add VPC endpoints for ECR, Secrets Manager, and CloudWatch so traffic to these services never leaves AWS's internal network (no internet exposure, faster, cheaper).

---

## 8. Is This Enterprise-Grade?

### Current state: **Production-ready for low-to-medium traffic**

| Capability | Current | Enterprise-grade |
|-----------|---------|-----------------|
| Zero-downtime deploys | ✓ ECS rolling replacement | ✓ Same |
| Health monitoring | ✓ ALB health checks + CloudWatch | + Alarms, PagerDuty, dashboards |
| Persistent storage | ✓ EFS | ✓ Same |
| Secret management | ✓ Secrets Manager | ✓ Same |
| Availability | Single task, single AZ | Multi-AZ, auto-scaling |
| HTTPS / custom domain | ❌ HTTP only | ✓ ACM + Route 53 |
| API authentication | ❌ Open | ✓ Cognito / API Gateway |
| Network isolation | Public subnet | Private subnet + NAT Gateway |
| Observability | CloudWatch Logs | + Traces (X-Ray), metrics, dashboards |
| DR / backup | EFS regional replication | Multi-region + automated backup |
| Auto-scaling | ❌ Fixed 1 task | ECS Service Auto Scaling |

The architecture **uses the same building blocks** as enterprise systems (Fargate, ALB, EFS, Secrets Manager, IAM, CloudWatch). The gaps are operational hardening rather than fundamental architectural flaws. Every missing piece above is addable without rebuilding from scratch.

---

## 9. Rough Monthly Cost Estimate

All prices are AWS us-east-1 on-demand rates as of 2026. Actual bills vary with usage.

### Fixed costs (running 24/7)

| Service | Calculation | Monthly cost |
|---------|------------|-------------|
| **ECS Fargate — CPU** | 2 vCPU × $0.04048/vCPU-hour × 730h | ~$59 |
| **ECS Fargate — Memory** | 16 GB × $0.004445/GB-hour × 730h | ~$52 |
| **ALB** | $0.0225/hour × 730h (fixed) | ~$16 |
| **EFS storage** | ~10 GB × $0.30/GB-month | ~$3 |
| **CloudWatch Logs** | ~2 GB/month × $0.50/GB ingested | ~$1 |
| **Secrets Manager** | 1 secret × $0.40/secret/month | ~$0.40 |
| **ECR storage** | ~2 GB × $0.10/GB-month | ~$0.20 |
| **Total fixed** | | **~$131/month** |

### Variable costs (usage-dependent)

| Service | Unit cost | Notes |
|---------|----------|-------|
| ALB LCU | ~$0.008/LCU-hour | Scales with request count and data processed |
| OpenAI embeddings | ~$0.13/million tokens | text-embedding-3-large |
| OpenAI GPT-4o (captioning + generation) | ~$2.50/million input tokens | Per-request cost |
| EFS reads/writes | ~$0.03/GB | Data transfer cost |
| Data transfer out | $0.09/GB first 10 TB | Responses sent to internet |

### Rough total scenarios

| Usage level | Description | Estimated monthly |
|-------------|------------|------------------|
| **Dev / demo** | 1 task, light usage, few PDFs | **~$135/month** |
| **Small production** | 1 task, 50 PDFs/day, 100 queries/day | **~$160/month** |
| **Medium production** | 2 tasks (HA), 500 PDFs/day, 1000 queries/day | **~$350/month** |

### Cost optimization levers

1. **Fargate Spot** — Run tasks on spare AWS capacity at 50-70% discount. Risk: tasks can be interrupted (30-second warning). Acceptable for non-critical workloads. Could cut the ~$111 compute cost to ~$40.

2. **Graviton (ARM64) Fargate** — 20% cheaper than x86 Fargate. Requires rebuilding the Docker image for ARM (`--platform linux/arm64`). All our Python/OpenAI code runs on ARM natively.

3. **Reduce Fargate memory** — After profiling real usage, you might find 12 GB is sufficient (saving ~$13/month).

4. **EFS Infrequent Access tier** — Files not accessed for 30 days automatically move to cheaper storage ($0.025/GB vs $0.30/GB). The Ollama model weights are read on first pull but rarely thereafter.

5. **Qdrant Cloud** — Replace the self-hosted Qdrant sidecar with Qdrant Cloud's free tier (1 GB, 1 collection). Removes Qdrant from the ECS task, potentially letting you drop to 8 GB RAM and saving ~$26/month.

---

## 10. Future Enhancements — The Production Roadmap

### Phase 1: Security Hardening (Immediate)

**10.1 HTTPS with a Custom Domain**

Currently HTTP only. To add HTTPS:
1. Register a domain in Route 53 (or bring your own).
2. Request a free TLS certificate from AWS Certificate Manager (ACM).
3. Add an HTTPS listener (port 443) to the ALB with the ACM certificate.
4. Add an HTTP → HTTPS redirect rule on port 80.
5. Point your Route 53 A record (alias) to the ALB DNS name.

Result: `https://api.yourdomain.com` with a valid TLS certificate. Cost: $0 for ACM certificate, ~$0.50/month for Route 53 hosted zone.

**10.2 API Authentication**

Add a simple API key middleware in FastAPI:

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(key: str = Security(api_key_header)):
    if key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
```

Or use AWS API Gateway in front of the ALB for OAuth2/JWT-based authentication.

**10.3 Custom VPC with Private Subnets**

Move ECS tasks into private subnets. Add a NAT Gateway for outbound internet (OpenAI API calls). This ensures containers have no public IP and are unreachable from the internet.

---

### Phase 2: Reliability and Availability

**10.4 Multi-AZ Deployment**

Run two tasks — one in `us-east-1a`, one in `us-east-1b`. The ALB automatically distributes requests between them. If one AZ loses power, the other handles all traffic.

Challenge: both tasks will share the same EFS volume. Qdrant supports concurrent reads from multiple instances but not concurrent writes. Solution: deploy Qdrant as a separate service with its own ALB (internal) and configure write-through from only one app instance.

**10.5 Auto Scaling**

```json
{
  "TargetTrackingScalingPolicy": {
    "TargetValue": 70.0,
    "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
  }
}
```

ECS Service Auto Scaling can add/remove tasks based on CPU utilization, memory, or custom CloudWatch metrics (e.g., ALB request count). When traffic spikes, new tasks spin up in 1-2 minutes.

**10.6 Circuit Breaker on ECS Service**

Enable ECS deployment circuit breaker. If a new task revision fails health checks repeatedly during a rollout, ECS automatically rolls back to the previous working revision. This prevents bad deployments from causing prolonged outages.

---

### Phase 3: Performance

**10.7 GPU-Enabled Instances for Ollama**

Currently, GLM-OCR runs on CPU inside Fargate. This is functional but slow (30-120 seconds per page). For production throughput, run Ollama on an EC2 GPU instance:

- `g4dn.xlarge` (NVIDIA T4, 16 GB GPU RAM): ~$0.52/hour
- `g5.xlarge` (NVIDIA A10G, 24 GB GPU RAM): ~$1.01/hour

GLM-OCR on a T4 GPU processes pages in 1-3 seconds vs 20-60 seconds on CPU.

Architecture change: extract Ollama into its own ECS service running on a GPU-enabled capacity provider (EC2 launch type, not Fargate), and have the app container call it over the VPC internal network.

**10.8 Model Caching — Pre-warm on Startup**

PP-DocLayoutV3 currently downloads from HuggingFace on first ingest request (cold start). Bake it into the Docker image at build time:

```dockerfile
RUN python -c "from transformers import AutoModelForObjectDetection; \
  AutoModelForObjectDetection.from_pretrained('PaddlePaddle/PP-DocLayoutV3_safetensors')"
```

This adds ~500 MB to the image but eliminates the HuggingFace download entirely. First ingest drops from 45 seconds to 5 seconds.

**10.9 Async Ingestion with SQS**

Currently, `/ingest/file` is synchronous — the HTTP connection stays open for the entire 2-5 minute pipeline. This works but is fragile (network interruptions fail the request).

Future architecture:
1. `POST /ingest/file` — saves the PDF to S3, puts a message on an SQS queue, immediately returns `202 Accepted` with a `job_id`.
2. A separate ECS worker task polls SQS and processes ingestion in the background.
3. `GET /ingest/status/{job_id}` — client polls for completion.

Result: the client connection closes in < 1 second. The pipeline runs for as long as needed without any timeout risk.

---

### Phase 4: Observability

**10.10 CloudWatch Dashboard**

Create a CloudWatch dashboard with:
- ECS task CPU and memory utilization
- ALB request count, latency (p50/p95/p99), 4xx/5xx error rates
- EFS throughput
- Custom metrics: ingest duration, chunks per document, query latency

**10.11 Distributed Tracing with AWS X-Ray**

Add X-Ray SDK to the FastAPI app. Every request gets a trace ID. You can visualize the entire request path: how long parsing took, how long embedding took, how long Qdrant search took. Invaluable for performance debugging.

**10.12 Alerting**

```
CloudWatch Alarm → SNS Topic → Email / PagerDuty / Slack
```

Critical alarms to add:
- ECS task count < desired (task replacement happening)
- ALB 5xx error rate > 1% over 5 minutes
- ECS memory utilization > 85% (approaching OOM)
- ALB p99 latency > 30 seconds

---

### Phase 5: Data and ML Enhancements

**10.13 S3 for Document Storage**

Store original PDFs in S3 after ingestion. Benefits:
- Re-ingest without re-uploading (just reference the S3 path)
- S3 event notifications → automatically trigger ingestion when files are uploaded to a bucket
- Long-term archival at ~$0.023/GB-month (far cheaper than EFS)

**10.14 Qdrant Collection Versioning**

Use collection naming conventions (`documents-v2`, `documents-v3`) to support:
- A/B testing different embedding models
- Blue-green deployment of the vector index (build new index in background, switch atomically)
- Rollback if new embedding model performs worse

**10.15 Fine-Tuned Embedding Model**

`text-embedding-3-large` is a general-purpose model. For domain-specific documents (medical, legal, financial), fine-tuning embeddings on your document corpus can improve retrieval precision by 10-30%.

**10.16 Streaming Responses**

Implement Server-Sent Events (SSE) on the `/generate` endpoint so GPT-4o tokens stream to the client in real time, rather than waiting for the full response. Dramatically improves perceived latency.

---

### Phase 6: CI/CD and GitOps

**10.17 GitHub Actions Pipeline**

```
git push → GitHub Actions:
  1. Run unit tests (pytest)
  2. Build Docker image (--platform linux/amd64)
  3. Push to ECR
  4. Register new ECS task definition
  5. Update ECS service
  6. Wait for stable
  7. Run smoke test (curl /health)
```

Credentials needed (in GitHub Secrets):
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` — for `doc-parser-cicd` IAM user (scoped to ECR push + ECS update only)

**10.18 Infrastructure as Code (Terraform or AWS CDK)**

Currently the infrastructure was created via CLI commands. Adopt Terraform or AWS CDK so the entire infrastructure is reproducible from code:

```bash
terraform apply   # creates VPC, ALB, ECS cluster, EFS, IAM roles, etc.
terraform destroy # tears it all down cleanly
```

This makes it trivial to replicate the environment for staging, regional expansion, or disaster recovery.

---

## 11. Key Architectural Trade-offs Made

| Decision | Trade-off accepted | Why it was the right call |
|----------|-------------------|--------------------------|
| Sidecar pattern (all containers in one task) | Can't scale Qdrant and Ollama independently | Simplicity and latency win at this scale |
| Default VPC / public subnets | Slightly less secure than private subnets | Faster to set up; security groups compensate |
| Synchronous ingest API | Long-lived HTTP connections, timeout risk | Simpler client code; fixed by 300s ALB timeout |
| CPU-only Ollama (no GPU) | Slow OCR (30-120s/page) | Fargate doesn't support GPU; acceptable for demo/low traffic |
| Single-task (no HA) | Single point of failure | Cost-appropriate for demo; HA is Phase 2 |
| `text-embedding-3-large` with 3072 dims | Higher cost and storage vs smaller models | Best retrieval quality for multi-modal content |
| EFS for Qdrant storage | EFS NFS latency slightly higher than local SSD | Persistence and rolling deployments require shared filesystem |

---

## 12. Glossary for Students

| Term | Plain-English Definition |
|------|--------------------------|
| **VPC** | Your private network inside AWS. Like a company's internal network, but in the cloud. |
| **Subnet** | A subdivision of the VPC. Public subnets have direct internet access; private subnets do not. |
| **Security Group** | A virtual firewall attached to a resource. Defines which traffic is allowed in and out. |
| **ALB** | A smart traffic director that receives requests from the internet and forwards them to your containers. |
| **ECS** | AWS's container orchestration system. Runs and manages Docker containers. |
| **Fargate** | ECS's serverless mode. You don't manage any servers — AWS handles the underlying compute. |
| **Task Definition** | A JSON blueprint describing which containers to run, how much CPU/RAM to give them, and what environment variables to inject. |
| **Service** | An ECS abstraction that keeps N copies of a task running at all times, replacing failed tasks automatically. |
| **ECR** | AWS's private Docker Hub. Stores your container images securely. |
| **EFS** | A managed network filesystem (like a shared network drive) that multiple containers can mount simultaneously. |
| **Secrets Manager** | A secure vault for API keys and passwords. Injects them into containers at runtime — never stored in code or files. |
| **IAM Role** | A set of permissions assigned to an AWS resource (not a person). The ECS task uses a role to call other AWS services. |
| **CloudWatch Logs** | AWS's centralized logging system. All container stdout/stderr goes here and is queryable. |
| **Health Check** | A periodic HTTP request the ALB makes to your container to verify it is alive and serving traffic correctly. |
| **Rolling Deployment** | A deployment strategy where new tasks start and pass health checks before old tasks are stopped. Results in zero downtime. |
| **Sidecar Pattern** | Running helper containers alongside your main app container in the same task, sharing localhost networking. |
| **Hybrid RAG** | Retrieval that combines dense vector search (semantic similarity) with sparse keyword search (BM25), then merges results using Reciprocal Rank Fusion (RRF). |
| **awsvpc** | The ECS network mode that gives each task its own private IP address inside the VPC. |
| **ECS Exec** | A feature that lets you open a shell inside a running container for live debugging, using AWS Systems Manager as the transport. |
| **NAT Gateway** | Allows resources in private subnets to make outbound internet calls (e.g., to OpenAI's API) without being reachable from the internet. |
| **Target Group** | The ALB's list of backend destinations (ECS tasks) it routes traffic to, plus health check configuration. |
| **ACM** | AWS Certificate Manager. Issues and renews free TLS certificates for HTTPS. |
