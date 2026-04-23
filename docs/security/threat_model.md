# Orchid Ranker Security Threat Model

## 1. Overview

This document outlines the security architecture and threat model for Orchid Ranker, a progression-aware recommender system library. It is intended for security reviewers, enterprise procurement teams, and system operators evaluating Orchid Ranker for production deployment.

Orchid Ranker is **not a service**—it is a Python library embedded within the customer's application. This document clarifies security boundaries, responsibilities, and residual risks.

## 2. System Boundaries

Orchid Ranker operates as a library within the customer's application runtime. The trust boundary is defined as follows:

```
┌─────────────────────────────────────────────────────────┐
│ Customer's Application                                  │
│  ┌─────────────────────────────────────────────────────┐│
│  │ Orchid Ranker Library (Library Code Below)           ││
│  │ • AccessControl (RBAC)                               ││
│  │ • JWTAuthenticator (optional)                        ││
│  │ • DP-SGD with RDP Accounting                         ││
│  │ • HMAC Hash-Chained Audit Logging                    ││
│  │ • Model Serialization & Checksums                    ││
│  │ • Connectors (Snowflake, BigQuery, S3, MLflow)       ││
│  │ • StudentAgent (simulation)                          ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
         │
         ├─ TLS/HTTPS to external services (operator responsibility)
         ├─ OS kernel, container runtime (operator responsibility)
         ├─ Identity & access management (operator responsibility)
         └─ Key management & encryption at rest (operator responsibility)
```

**Key assumption**: The customer's application runtime and infrastructure are under the operator's control. Orchid Ranker provides library-level controls; infrastructure hardening is the operator's responsibility.

## 3. What Orchid Protects

### 3.1 Data Access Control
- **RBAC via AccessControl**: Four roles (admin, ml_engineer, analyst, viewer) with action-based permissions
- **Optional JWT/OIDC**: Provider-agnostic JWT authentication with role extraction and signature validation
- **Policy enforcement**: `AccessControl.require()` raises `PermissionError` on unauthorized access

### 3.2 User Privacy
- **Differential Privacy (DP-SGD)**: Adds calibrated noise to gradient updates during training
- **RDP Accounting**: Opacus-based Rényi differential privacy accounting tracks cumulative privacy budget
- **Configurable epsilon budgets**: Operators set epsilon/delta trade-off per training regime
- **Privacy guarantees**: (ε, δ)-differential privacy means aggregate output depends minimally on any individual record

### 3.3 Audit Trail Integrity
- **HMAC hash-chained logging**: Sequential logs are cryptographically bound; tampering is detectable
- **Optional Fernet encryption at rest**: Logs can be encrypted with symmetric keys before storage
- **Immutability via chaining**: Each log entry includes HMAC of previous entry, making retroactive modification impossible without invalidating the chain

### 3.4 Model Integrity
- **Serialization with checksums**: Models saved with `torch.save()` and version metadata
- **Format validation on load**: Checkpoint structure (version, model_type, state) is validated before restoration
- **Type checking**: `load_model()` rejects unknown model types; supported types are whitelisted

### 3.5 Credential Security
- **from_env/from_vault patterns**: Connectors (Snowflake, BigQuery, S3, MLflow) support credential loading from environment or secret vaults
- **Password masking**: Sensitive fields (e.g., passwords) are masked in `__repr__()` to prevent accidental log leakage
- **No hardcoded secrets**: Library design encourages injecting credentials at runtime rather than embedding them

## 4. What Orchid Delegates (Operator Responsibility)

| Domain | Responsibility | Examples |
|--------|---|---|
| **Network Security** | TLS termination, firewall rules, DDoS mitigation | Ensure HTTPS for all external API calls, restrict egress to known services |
| **Infrastructure Hardening** | OS patching, container image scanning, runtime security | Apply OS patches, use secure base images, enforce seccomp/AppArmor |
| **Identity Management** | IdP configuration, user provisioning, revocation | Configure OIDC provider (Okta, Auth0, etc.), manage user lifecycle |
| **Key Management** | HMAC key storage, encryption key rotation, JWT key lifecycle | Rotate keys regularly, store in HSM or KMS, enforce access policies |
| **Data at Rest Encryption** | Disk-level encryption for databases, file systems, backups | Use full-disk encryption, encrypted block storage, encrypted backups |
| **Backup & Disaster Recovery** | Secure backup storage, version control, integrity checks | Encrypt backups, test restore procedures, verify checksum integrity |
| **Logging & Monitoring** | Centralized log aggregation, alerting, log retention | Forward audit logs to SIEM, set alerts for privilege escalation, retain per policy |

## 5. Threat Matrix

| # | Threat | Category | Mitigation | Residual Risk | Severity |
|---|--------|----------|-----------|---|----------|
| 1 | **Model poisoning via malicious training data** | Tampering | Validate input data schema and distributions; log all training dataset accesses; use differential privacy to limit single-record influence | Distributed attacks or subtle poisoning may evade detection if DP epsilon budget is high | Medium |
| 2 | **Deserialization attacks (torch.load pickle gadgets)** | Tampering | Validate checkpoint format and model_type before instantiation; do not deserialize untrusted checkpoints; use code review for custom model classes | Operators may still deserialize untrusted files if not trained on safe practices | High |
| 3 | **Privacy budget exhaustion without user awareness** | Information Disclosure | RDP accountant tracks epsilon; library warns when budget near exhaustion; deploy monitoring on epsilon drift | Attackers may conduct membership inference via repeated queries if epsilon budget exhausted silently | Medium |
| 4 | **Audit log tampering via replay or forgery** | Repudiation | HMAC hash-chaining detects tampering; optional Fernet encryption protects at rest | Logs in transit are unencrypted unless TLS enforced by operator; keys must be protected | Medium |
| 5 | **Credential leakage (API keys, passwords)** | Information Disclosure | `from_env()`/`from_vault()` patterns; password masking in logs; no default hardcoding | Developer mistakes (e.g., logging credentials directly) still possible; requires discipline | Medium |
| 6 | **Privilege escalation via role manipulation** | Elevation of Privilege | RBAC policy is immutable dataclass; AccessControl validates role/action at enforcement points | Operator misconfiguration (e.g., assigning admin role too broadly) not prevented | Medium |
| 7 | **Side-channel attacks on DP noise generation** | Information Disclosure | Opacus uses cryptographically secure RNG (torch.randn w/ seed); timing of noise addition is constant per batch | Timing attacks on gradient clipping or noise generation theoretically possible but require fine-grained measurement | Low |
| 8 | **Replay attacks on JWT tokens** | Spoofing | JWTAuthenticator validates exp (expiration) claim; operator must enforce short token TTLs | Tokens valid until expiry can be replayed; no built-in revocation list; requires operator infrastructure | Medium |
| 9 | **SQL injection via connector parameterization (Snowflake/BigQuery)** | Tampering | Connectors accept parameterized queries; ORM-like patterns prevent raw SQL concatenation | Operators using raw SQL templates may introduce injection; requires query review | Medium |
| 10 | **Denial of service via large model files** | Denial of Service | No size limits enforced on `load_model()`; operator must set filesystem quotas and network bandwidth limits | Memory exhaustion or disk full possible if untrusted checkpoint loaded; requires monitoring | Low |
| 11 | **Unauthorized access to sensitive actions (e.g., dp_sensitive)** | Elevation of Privilege | `AccessControl.require()` enforces policy; missing checks bypass protection | Developers may forget to call `require()` before sensitive operations | Medium |
| 12 | **Composition attacks across DP training sessions** | Information Disclosure | RDP accounting is per-accountant instance; composition across independent sessions is operator responsibility | Attackers may combine inferences from multiple training runs; operators must manage overall privacy budget | Medium |

## 6. Serialization Security

### Attack Surface

PyTorch's `torch.load()` uses pickle, which is known to execute arbitrary code during deserialization:

```python
# Unsafe pattern (VULNERABLE):
model = torch.load(untrusted_checkpoint)  # May execute arbitrary code
```

### Mitigations Implemented

1. **Whitelist model types**: `load_model()` only accepts `OrchidRecommender` or `TwoTowerRecommender`
2. **Validate checkpoint structure**: Missing `version`, `model_type`, or `state` keys raise `ValueError`
3. **Version checking**: Mismatched versions trigger a warning
4. **Type-specific restoration**: Each model type has a dedicated restoration function with explicit construction

```python
# Safe pattern (IMPLEMENTED):
if model_type not in {"OrchidRecommender", "TwoTowerRecommender"}:
    raise ValueError(f"Unknown model type: {model_type}")
```

### Recommendations for Deployers

- **Checksum verification**: Before loading, verify checksum of checkpoint file (e.g., SHA-256)
- **Checkpoint provenance**: Only load checkpoints from trusted sources (your own model zoo, signed releases)
- **Filesystem quotas**: Set ulimits or container resource limits to prevent disk exhaustion
- **Sandboxing**: Consider loading models in a separate, isolated process if checkpoint source is semi-trusted
- **No pickle in untrusted environments**: If checkpoint source is untrusted, use alternative serialization (e.g., ONNX, SafeTensors)

## 7. Privacy Guarantees

### What (ε, δ)-Differential Privacy Means

Differential privacy guarantees that the presence or absence of any single individual in the training dataset has minimal impact on the model's output:

- **ε (epsilon)**: Privacy loss budget. Smaller ε = stronger privacy. Typical values: 1.0 (strong), 8.0 (moderate)
- **δ (delta)**: Failure probability. Typical values: 1e-5 to 1e-6 (1 in 100k to 1 million)
- **Interpretation**: For any query result, an adversary cannot distinguish whether a particular individual was included in training with probability better than e^ε (plus δ failure)

### How RDP Accounting Works

Orchid uses **Rényi Differential Privacy (RDP)** accounting via Opacus:

1. **Per-batch noise**: Each training step clips gradients (max_grad_norm) and adds Gaussian noise (σ = noise_multiplier)
2. **RDP composition**: The accountant computes RDP values for each order λ ∈ {1.25, 1.5, 2, ...} and accumulates across steps
3. **Epsilon conversion**: Uses Opacus's `get_privacy_spent()` to convert RDP to (ε, δ)-DP

```python
accountant = OpacusAccountant(
    sample_rate=0.01,          # Batch size / dataset size
    noise_multiplier=1.0,       # Relative noise level
    delta=1e-5,                # Failure probability
)
eps_consumed, total_eps = accountant.step(num_training_steps)
```

### Privacy Budget Exhaustion

When cumulative ε exceeds the configured budget:
- RDP accountant continues to return increasing ε values
- **No automatic stopping**: Operator must monitor and decide whether to halt training
- **Composition across sessions**: Each independent DP training session increments ε; operators must track aggregate privacy cost

### Limitations

1. **No revocation**: Once data is used in DP training, privacy cost is permanent
2. **Composition across independent sessions**: Training on dataset A with ε=1 then on dataset B with ε=1 yields ε=2 total for either dataset
3. **Loose bounds**: RDP upper bounds may be conservative, yielding ε estimates higher than necessary
4. **No subsampling guarantees for custom optimizers**: DP-SGD assumes Poisson subsampling; other patterns may degrade guarantees

## 8. Recommendations for Deployers

### Pre-Production Checklist

- [ ] **Enable RBAC**: Configure AccessControl policy matching your role hierarchy; review default policy (admin, ml_engineer, analyst, viewer)
- [ ] **Enable JWT authentication**: If using JWTAuthenticator, configure issuer, audience, and role_claim to match your OIDC provider; test token validation
- [ ] **Secure credential storage**: Use `from_env()` / `from_vault()` for all connectors; never embed passwords in code; rotate credentials regularly
- [ ] **Protect HMAC keys**: Store audit log HMAC keys in HSM or managed key store; rotate monthly
- [ ] **Enable Fernet encryption**: If storing audit logs on disk, enable optional Fernet encryption at rest
- [ ] **Configure DP budgets**: Define epsilon/delta targets for your use case; set monitoring alerts when ε approaches budget
- [ ] **Validate model checksums**: Before deploying models, verify checksums; store checksums separately from model files
- [ ] **Set up centralized logging**: Forward all Orchid logs (including audit logs) to SIEM; configure alerts for privilege escalation attempts
- [ ] **Test backup/recovery**: Ensure audit logs and models are backed up securely; test restore procedure
- [ ] **Review connector configurations**: Audit all data source credentials and network policies; ensure TLS is enforced for remote connections
- [ ] **Network policy**: Restrict egress to known services (Snowflake, BigQuery, S3, MLflow); use firewall rules or service mesh
- [ ] **Incident response**: Document incident response procedures for credential compromise, audit log tampering, or privacy budget overruns

---

**Document Version**: 1.0  
**Last Updated**: 2026-04-09  
**Audience**: Security reviewers, enterprise procurement, system operators
