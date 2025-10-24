# Data Retention Policy (Draft)

## Scope
- Applies to all Orchid Ranker generated artefacts: preprocessed datasets, experiment outputs, audit logs, SBOMs, and DP telemetry.

## Retention Windows
- Raw customer data: retained only for the duration of preprocessing (ephemeral storage) unless contractually specified; default 30 days then purged.
- Derived datasets (normalized CSV bundles): retained 90 days for reproducibility, then archived or deleted per customer preference.
- Experiment logs & audit trails: retained 12 months to satisfy compliance inquiries.
- SBOMs and vulnerability reports: retained 24 months to support security audits.

## Storage & Encryption
- All persistent artefacts must reside in encrypted storage (e.g., AWS S3 with SSE, GCS CMEK).
- Access controlled via IAM roles mapped to Orchid RBAC roles (admin/ml_engineer/analyst/viewer).

## Deletion Procedures
- Automated lifecycle policies purge objects after retention window.
- Manual deletion requests must be honored within 7 business days; document request and completion in the audit log.

## Backups
- Backups of experiment results and audit logs should follow the same retention limits. Backups may not extend data lifetime beyond policy limits.

## Review Cadence
- Policy reviewed annually or upon material change in regulatory requirements.

## Responsibilities
- Product Manager: ensure customers are informed of retention defaults.
- DevOps: configure lifecycle policies and monitor enforcement.
- Security Lead: verify retention compliance during quarterly audits.
