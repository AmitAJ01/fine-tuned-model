"""Dataset preparation and seed generator script.

Prepares curated DevOps incident training and evaluation splits adhering to
the FineTuneOps structured diagnosis schema.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, List
from ml.training.dataset import DevOpsIncidentDataset, prepare_data_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finetuneops.dataset")

# Curated high-fidelity DevOps incident seed examples representing real-world production outages
CURATED_INCIDENTS: List[Dict[str, Any]] = [
    {
        "id": "inc-001",
        "log_text": """
2026-10-31T14:10:02Z kernel: [129482.102] Out of memory: Killed process 8421 (java) total-vm:18942000kB, anon-rss:14920000kB, file-rss:0kB
2026-10-31T14:10:03Z kubelet: container billing-service failed liveness probe, will be restarted
2026-10-31T14:10:04Z kubelet: OOMKilled: Pod billing-service-7f89c4d9b-x98pk exceeded memory limit (16Gi)
2026-10-31T14:10:05Z envoy: [downstream: billing-service] 503 Service Unavailable, reset reason: connection termination
""",
        "system_context": "Kubernetes 1.30, EKS on AWS, Java 21 Spring Boot Microservice.",
        "target_report": {
            "incident_category": "Infrastructure",
            "severity_level": "SEV-1",
            "root_cause_analysis": "The billing-service JVM container exceeded the 16Gi memory limit specified in its Kubernetes resource quota, triggering the Linux kernel OOM-killer.",
            "contributing_factors": [
                "Unbounded in-memory queuing during end-of-month batch transaction settlement",
                "JVM MaxRAMPercentage not set, causing Java heap to size based on node rather than cgroup limits",
            ],
            "recommended_debugging_steps": [
                "kubectl describe pod billing-service-7f89c4d9b-x98pk | grep -A 5 -i oom",
                "Inspect Prometheus container_memory_working_set_bytes metric leading up to 14:10:02Z",
                "Capture JVM heap dump analysis using Eclipse Memory Analyzer (MAT)",
            ],
            "suggested_remediation_actions": [
                "Configure JVM flags: -XX:InitialRAMPercentage=70.0 -XX:MaxRAMPercentage=75.0",
                "Implement backpressure or offload batch records to Amazon SQS",
                "Increase Kubernetes memory request/limit to 24Gi with horizontal pod autoscaler (HPA)",
            ],
        },
    },
    {
        "id": "inc-002",
        "log_text": """
2026-10-31T15:22:11.450Z [ERROR] [pool-1-thread-44] org.postgresql.util.PSQLException: FATAL: remaining connection slots are reserved for non-replication superuser connections
2026-10-31T15:22:11.452Z [ERROR] HikariPool-1 - Connection is not available, request timed out after 30005ms.
2026-10-31T15:22:12.010Z [WARN] OrderService: Failed to persist order 992482: timeout waiting for physical connection
2026-10-31T15:22:13.110Z [FATAL] HealthCheck: Database health check failed: database ping timeout (timeout 5s)
""",
        "system_context": "PostgreSQL 16 RDS Aurora, 12 FastAPI and Spring application replicas.",
        "target_report": {
            "incident_category": "Database",
            "severity_level": "SEV-2",
            "root_cause_analysis": "PostgreSQL connection pool exhaustion caused by client connection leaks across scaled worker replicas, reaching max_connections (300).",
            "contributing_factors": [
                "Auto-scaling group added 8 new application pods, multiplying idle connection pool sizes",
                "Absence of PgBouncer connection multiplexing proxy between application pods and Aurora",
            ],
            "recommended_debugging_steps": [
                "SELECT count(*), state, usename FROM pg_stat_activity GROUP BY state, usename;",
                "Check HikariCP active vs idle connection metrics in Datadog/Grafana",
                "Identify longest-running open transactions: SELECT pid, query, age(clock_timestamp(), query_start) FROM pg_stat_activity WHERE state != 'idle';",
            ],
            "suggested_remediation_actions": [
                "Terminate leaked idle connections using pg_terminate_backend()",
                "Deploy AWS RDS Proxy or PgBouncer in transaction pooling mode",
                "Reduce HikariCP maximumPoolSize from 30 down to 10 per application container",
            ],
        },
    },
    {
        "id": "inc-003",
        "log_text": """
2026-10-31T08:04:19Z kubelet: Pulling image "registry.internal.corp/checkout-api:v2.4.1"
2026-10-31T08:04:20Z kubelet: Successfully pulled image in 1.12s
2026-10-31T08:04:22Z kubelet: Created container checkout-api
2026-10-31T08:04:23Z kubelet: Started container checkout-api
2026-10-31T08:04:24Z checkout-api: [Error: ENOENT: no such file or directory, open '/etc/secrets/stripe_private_key.pem']
2026-10-31T08:04:24Z checkout-api: Process exited with code 1
2026-10-31T08:04:25Z kubelet: Back-off restarting failed container checkout-api in pod checkout-api-6874fb6d7d-b5z6d (CrashLoopBackOff)
""",
        "system_context": "Kubernetes cluster with HashiCorp Vault secrets CSI driver.",
        "target_report": {
            "incident_category": "Kubernetes / Container Orchestration",
            "severity_level": "SEV-1",
            "root_cause_analysis": "The checkout-api container entered CrashLoopBackOff on startup due to missing secret mount '/etc/secrets/stripe_private_key.pem'.",
            "contributing_factors": [
                "Recent Helm deployment referenced an outdated SecretProviderClass key name",
                "Lack of admission webhook validation verifying secret existence before rollout",
            ],
            "recommended_debugging_steps": [
                "kubectl describe pod checkout-api-6874fb6d7d-b5z6d",
                "kubectl get secretproviderclass -n checkout -o yaml",
                "kubectl logs -l app=secrets-store-csi-driver -n kube-system",
            ],
            "suggested_remediation_actions": [
                "Roll back the Helm release to previous stable revision: helm rollback checkout-api",
                "Verify and update SecretProviderClass vault mapping for stripe_private_key.pem",
                "Add pre-flight config validation in CI/CD pipeline",
            ],
        },
    },
    {
        "id": "inc-004",
        "log_text": """
2026-10-31T11:45:00Z traefik: 504 Gateway Timeout GET /api/v1/recommendations duration=30001ms
2026-10-31T11:45:01Z core-dns: [WARNING] plugin/forward: no nameservers found
2026-10-31T11:45:02Z core-dns: [ERROR] 2 rec-engine.service.internal. A: dial udp 10.96.0.10:53: i/o timeout
2026-10-31T11:45:04Z recommendations: lookup rec-engine.service.internal on 10.96.0.10:53: read udp 10.244.1.18:54212->10.96.0.10:53: i/o timeout
""",
        "system_context": "CoreDNS on Kubernetes, Calico CNI.",
        "target_report": {
            "incident_category": "Network",
            "severity_level": "SEV-2",
            "root_cause_analysis": "Internal DNS resolution failure in CoreDNS pods leading to 504 gateway timeouts across microservices.",
            "contributing_factors": [
                "Calico node network policy dropped UDP port 53 packets between worker nodes",
                "CoreDNS replica count (2) was insufficient for cluster query volume",
            ],
            "recommended_debugging_steps": [
                "kubectl get pods -n kube-system -l k8s-app=kube-dns -o wide",
                "kubectl exec -it netshoot -- dig @10.96.0.10 rec-engine.service.internal",
                "Check CoreDNS logs: kubectl logs -n kube-system -l k8s-app=kube-dns",
            ],
            "suggested_remediation_actions": [
                "Scale CoreDNS deployment to at least 4 replicas with NodeLocal DNSCache",
                "Verify and patch Calico CNI policy allowing inter-node UDP/TCP 53 traffic",
                "Monitor coredns_dns_request_duration_seconds in Prometheus",
            ],
        },
    },
    {
        "id": "inc-005",
        "log_text": """
2026-10-31T03:15:20Z redis-server[1]: # WARNING: Redis is configured to save RDB snapshots, but is currently not able to persist on disk. Commands that may modify the data set are disabled. Error: MISCONF Errors writing to the append-only file: No space left on device
2026-10-31T03:15:21Z session-mgr: redis.exceptions.ResponseError: MISCONF Errors writing to the append-only file: No space left on device
2026-10-31T03:15:22Z auth-api: 500 Internal Server Error: Failed to write user session token to Redis
""",
        "system_context": "Redis 7.2 Cluster on Linux EBS gp3 volume.",
        "target_report": {
            "incident_category": "Infrastructure",
            "severity_level": "SEV-1",
            "root_cause_analysis": "Redis persistence failure caused by root disk volume exhaustion (100% full), forcing Redis into read-only protective mode.",
            "contributing_factors": [
                "AOF (Append Only File) auto-rewrite was disabled, allowing appendonly.aof to consume 100GB",
                "Disk alert threshold of 90% was unmonitored during the night window",
            ],
            "recommended_debugging_steps": [
                "df -h /var/lib/redis",
                "ls -lhS /var/lib/redis",
                "redis-cli info persistence",
            ],
            "suggested_remediation_actions": [
                "Trigger online volume expansion on AWS EBS volume (/dev/xvdf)",
                "Execute BGREWRITEAOF manually once disk headroom is restored",
                "Configure redis.conf auto-aof-rewrite-percentage 100 and disk monitoring alerts in PagerDuty",
            ],
        },
    },
]


def prepare_dataset_artifacts() -> None:
    """Prepare raw data and train/val/test splits."""
    raw_dir = Path("ml/data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / "incidents_seed.jsonl"

    logger.info(f"Writing {len(CURATED_INCIDENTS)} curated incident records to {raw_file}")
    with open(raw_file, "w", encoding="utf-8") as f:
        for inc in CURATED_INCIDENTS:
            f.write(json.dumps(inc, ensure_ascii=False) + "\n")

    # Format into SFT chat messages
    sft_records = []
    for inc in CURATED_INCIDENTS:
        formatted = DevOpsIncidentDataset.format_sample(
            log_text=inc["log_text"],
            target_report=inc["target_report"],
            system_context=inc.get("system_context"),
        )
        # Retain original fields for benchmark evaluation
        formatted["id"] = inc["id"]
        formatted["input_log"] = inc["log_text"]
        formatted["ground_truth"] = inc["target_report"]
        sft_records.append(formatted)

    # Save to splits (train, val, test)
    prepare_data_splits(
        sft_records,
        output_dir="ml/data/splits",
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
    )
    logger.info("Dataset preparation complete.")


if __name__ == "__main__":
    prepare_dataset_artifacts()
