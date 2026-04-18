# Multi-AZ Subnet Architecture

## Overview

The deployment uses **4 subnets across 2 Availability Zones** inside a single VPC (`doc-parser-vpc`, `10.0.0.0/16`). Each AZ gets one public subnet and one private subnet — this is the standard "dual-subnet" pattern for production AWS workloads.

| Subnet | CIDR | AZ | Tier |
|--------|------|----|------|
| Public Subnet A | 10.0.1.0/24 | us-east-1a | Public |
| Private Subnet A | 10.0.2.0/24 | us-east-1a | Private |
| Public Subnet B | 10.0.3.0/24 | us-east-1b | Public |
| Private Subnet B | 10.0.4.0/24 | us-east-1b | Private |

---

## Why Public AND Private Subnets?

### Public Subnet — What lives here

| Resource | Why public? |
|----------|------------|
| Application Load Balancer (ALB) | Must be reachable from the internet to accept HTTPS traffic |
| NAT Gateway | Needs an Elastic IP and a route to the Internet Gateway |

The ALB is the **only entry point** into the system. Everything else sits behind it.

### Private Subnet — What lives here

| Resource | Why private? |
|----------|------------|
| ECS Fargate task (app, qdrant, ollama) | Containers should never be directly reachable from the internet |
| EFS mount targets | Storage layer — no reason to expose externally |

Private subnets have **no route to the Internet Gateway**. Outbound internet traffic (e.g., OpenAI API calls) exits through the NAT Gateway in the public subnet. Inbound traffic only arrives via the ALB → target group path.

### The Flow

```
Internet
    │
    ▼
[Internet Gateway]
    │
    ▼
[ALB]  ←── lives in Public Subnet A + B
    │
    ▼  (target group forwards to port 8000)
[Fargate Task]  ←── lives in Private Subnet A (or B on failover)
    │
    ├──▶ Qdrant  (localhost:6333)
    ├──▶ Ollama  (localhost:11434)
    └──▶ [NAT Gateway] → OpenAI API / Z.AI API (outbound only)
```

---

## Why 2 AZs?

### Reason 1 — ALB requires it

AWS enforces that an Application Load Balancer must be associated with **at least 2 subnets in different Availability Zones** at creation time. You cannot create an ALB with a single subnet. This means:
- Public Subnet A → ALB node active in us-east-1a
- Public Subnet B → ALB node active in us-east-1b

Both ALB nodes share the same DNS name. Route 53 / AWS health checks route traffic to whichever node is healthy.

### Reason 2 — ECS Fargate failover

The ECS service is configured with both private subnets. ECS uses this to:

1. **Place the task** — on first launch, ECS picks whichever subnet has capacity
2. **Restart on failure** — if the Fargate task crashes and AZ-A is degraded, ECS restarts the replacement task in Private Subnet B (us-east-1b) automatically
3. **Scale out** — if you increase the desired count above 1, ECS spreads tasks across both AZs for redundancy

### Reason 3 — EFS multi-AZ access

EFS creates a **mount target in each AZ**. The Fargate task mounts the same EFS volume regardless of which AZ it runs in. There is no data duplication — both mount targets point to the same underlying file system. This means:

- Qdrant's vector data persists across task restarts in either AZ
- No manual data migration needed if the task moves between AZs

---

## Security — How Traffic Is Controlled

### Security Groups

| Security Group | Attached to | Inbound | Outbound |
|---------------|-------------|---------|----------|
| `sg-alb` | ALB | 443 from `0.0.0.0/0` | All to `sg-app` |
| `sg-app` | Fargate task | 8000 from `sg-alb` only | All (for OpenAI, Z.AI) |
| `sg-efs` | EFS mount targets | 2049 (NFS) from `sg-app` only | None |

The Fargate task is unreachable from the internet — its security group only accepts connections from the ALB's security group. Even if someone knew the private IP of the container, they could not reach it.

### Subnet Route Tables

| Subnet type | Route | Via |
|-------------|-------|-----|
| Public | `0.0.0.0/0` | Internet Gateway |
| Private | `0.0.0.0/0` | NAT Gateway (in public subnet) |
| Private | `10.0.0.0/16` | Local (VPC internal) |

Private subnets cannot receive inbound connections from the internet because there is no route from the Internet Gateway to them.

---

## Visual Reference

See [`multi-az-subnet-architecture.drawio`](./multi-az-subnet-architecture.drawio) for the full diagram showing both AZs, subnet tiers, and traffic flow.

---

## Summary

| Decision | Reason |
|----------|--------|
| Public subnet for ALB | ALB must be internet-facing |
| Private subnet for Fargate | Containers should not be directly reachable |
| 2 public subnets (one per AZ) | AWS ALB requirement — minimum 2 AZs |
| 2 private subnets (one per AZ) | ECS failover — task can restart in either AZ |
| NAT Gateway in public subnet | Private containers need outbound internet for API calls |
| EFS mount target in each AZ | Same data volume accessible regardless of which AZ runs the task |
