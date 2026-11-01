# Dataset Card for Cloud-OpsBench (FineTuneOps Integration)

## 1. Dataset Summary

**FineTuneOps** integrates the official public **Cloud-OpsBench** benchmark dataset to train and evaluate `Qwen2.5-Coder-1.5B-Instruct` on Kubernetes and cloud-native microservice incident diagnosis.

Cloud-OpsBench is a reproducible benchmark and infrastructure designed by **LLM4Ops** for evaluating agentic Root Cause Analysis (RCA) in cloud-native systems. Each incident is a preserved digital-twin state snapshot containing recorded control-plane telemetry, Kubernetes resource states, pod events, container error logs, and Prometheus alert anomalies across real microservice deployments (*Online Boutique* and *TrainTicket*).

- **Dataset Identifier:** `LLM4Ops/Cloud-OpsBench`
- **Distribution Version:** `v1.0.0`
- **Source Git Commit:** `ea05dafbc87b5985df8721622d63d4229c2b4224`
- **Official Benchmark Repo:** [https://github.com/LLM4Ops/Cloud-OpsBench](https://github.com/LLM4Ops/Cloud-OpsBench)
- **Hugging Face Hub:** [https://huggingface.co/datasets/LLM4Ops/Cloud-OpsBench](https://huggingface.co/datasets/LLM4Ops/Cloud-OpsBench)
- **Associated Paper:** *"Cloud-OpsBench: A Reproducible Benchmark for Agentic Root Cause Analysis in Cloud Systems"* (arXiv:2603.00468)
- **License:** MIT License

---

## 2. Dataset Structure & Schema

The dataset is formatted into instruction-response pairs adhering strictly to information present in Cloud-OpsBench.

### Record Schema
Each processed record in `ml/data/processed/incidents.jsonl` contains:

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `str` | Canonical case key (e.g. `boutique/admission/1`, `trainticket/runtime/12`) |
| `category` | `str` | Canonical category (`Kubernetes / Container Orchestration`, `Application`, `Infrastructure`, `Network`) |
| `severity` | `str` | Standardized severity (`Critical`, `High`, `Medium`, `Low`) mapped from benchmark difficulty and fault impact |
| `incident_description` | `str` | Initial incident query and system context (e.g., target namespace, microservice application) |
| `logs` | `str` | Observed diagnostic telemetry: Prometheus alerts, `kubectl get pods`, `kubectl describe`, events, container error logs |
| `root_cause` | `str` | Precise ground-truth failure mechanism (e.g., `missing_service_account`, `node_selector_mismatch`) |
| `contributing_factors` | `List[str]` | Factually supported architectural and environmental factors |
| `debugging_steps` | `List[str]` | Diagnostic tool calls executed during the investigation (e.g., `GetResources`, `DescribeResource`, `GetErrorLogs`) |
| `recommended_actions` | `List[str]` | Explicit disclaimer documenting benchmark scope (RCA-focused) |
| `expected_output` | `dict` | Structured canonical payload conforming to `IncidentReportPayload` |
| `source_type` | `str` | `"cloud_ops_bench_public"` |
| `source_reference` | `str` | Case provenance citation (e.g., `Cloud-OpsBench v1.0.0 (boutique/admission/1)`) |

---

## 3. Scope & Target Applications

The dataset covers two complex distributed microservice systems deployed on Kubernetes:

1. **Online Boutique (550 cases / 72.9%):** Google Cloud's 11-tier microservice e-commerce application written in Go, Java, Python, C#, and Node.js with Redis cache and gRPC communication.
2. **TrainTicket (204 cases / 27.1%):** High-concurrency 40-tier microservice train ticketing application written in Java Spring Cloud, Node.js, and Python with MySQL and MongoDB databases.

---

## 4. Fault Taxonomies & Root Cause Distribution

Cloud-OpsBench categorizes faults across eight distinct cloud-native failure classes:

| Fault Taxonomy | Case Count | % of Dataset | Mapped Canonical Category |
| :--- | :--- | :--- | :--- |
| **Scheduling_Fault** | 164 | 21.8% | Kubernetes / Container Orchestration |
| **Runtime_Fault** | 141 | 18.7% | Application |
| **Application_Code_Defect** | 98 | 13.0% | Application |
| **Service_Routing_Fault** | 91 | 12.1% | Network |
| **Startup_Fault** | 86 | 11.4% | Kubernetes / Container Orchestration |
| **Performance_Fault** | 76 | 10.1% | Infrastructure |
| **Admission_Fault** | 58 | 7.7% | Kubernetes / Container Orchestration |
| **Infrastructure_Fault** | 40 | 5.3% | Infrastructure |
| **Total** | **754** | **100.0%** | |

### Unique Root Causes (57 Total)
Top root causes include:
- `pod_network_delay` (57 cases)
- `node_selector_mismatch` (32 cases)
- `incorrect_image_reference` (28 cases)
- `taint_toleration_mismatch` (24 cases)
- `node_affinity_mismatch` (24 cases)
- `service_selector_mismatch` (24 cases)
- `mysql_invalid_credentials` (24 cases)
- `missing_secret_binding` (24 cases)
- `mysql_invalid_port` (22 cases)
- `code_missing_parameter` (20 cases)

---

## 5. Splitting Strategy & Data Leakage Prevention

- **Deterministic Split:** 70% Training (~528 cases), 15% Validation (~113 cases), 15% Testing (~113 cases) using random seed 42.
- **Group-Stratified Partitioning:** Cases sharing the same underlying fault scenario key `(system::category::fault_object::root_cause)` are assigned atomically to a single split. Zero scenario groups cross split boundaries.
- **Input Isolation (Zero Target Leakage):** The model input contains only the symptom query and raw diagnostic telemetry. Ground truth fields (`fault_taxonomy`, `fault_object`, `root_cause`, and milestone outcomes) are strictly excluded from the prompt.
- **Held-Out Test Set:** The test partition (`ml/data/splits/test.jsonl`) is strictly locked and reserved exclusively for post-fine-tuning evaluation.

---

## 6. Known Dataset Limitations

1. **Remediation Ground Truth:** Cloud-OpsBench is designed primarily as a Root Cause Analysis (RCA) benchmark. It evaluates whether an agent can locate the faulty component and identify the root cause mechanism. It does not provide benchmark ground-truth remediation scripts. Consequently, FineTuneOps explicitly disclaims remediation actions rather than fabricating ungrounded recommendations.
2. **Domain Focus:** The dataset focuses exclusively on Kubernetes and cloud-native microservices. It does not cover bare-metal OS, monolithic on-premises databases, or traditional hardware networking.
