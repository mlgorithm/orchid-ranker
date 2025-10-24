# FERPA & GDPR Alignment (Draft)

## Data Minimization
- Preprocessing pipelines should ingest only required learner identifiers and assessment data.
- Optional features (e.g., demographic fields) must be configurable and disabled by default unless explicit consent is obtained.

## Consent & Lawful Basis
- Platforms embedding Orchid must capture learner consent for data usage and clearly surface adaptive learning objectives.
- Maintain records of consent per customer tenancy.

## Student Rights (FERPA)
- Provide mechanisms for students/guardians to request access or deletion of records processed by Orchid pipelines.
- Data exports must be provided within 45 days of request; align deletion with retention policy.

## Data Subject Rights (GDPR)
- Support Right to Access, Rectification, and Erasure through API/CLI workflows.
- Ensure DP audit logs include tenant identifiers to facilitate compliance attestation.

## Cross-Border Transfers
- Default hosting regions should match customer residency. Document sub-processors and implement SCCs where applicable.

## Privacy-by-Design Controls
- Differential privacy engines (per-sample, Opacus) provide configurable epsilon/delta bounds.
- Audit logs capture DP parameters and access events to evidence compliance.
- RBAC enforcement limits preprocessing and experiment operations to authorized roles.

## Incident Notification
- Align with incident response playbook: notify educational institutions promptly (FERPA) and supervisory authorities within 72 hours when required (GDPR Article 33).

## Continuous Review
- Revisit alignment quarterly and upon regulatory updates.
- Maintain traceability from product features to compliance controls in the enterprise plan.
