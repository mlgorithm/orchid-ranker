# Incident Response Playbook (Draft)

## Purpose
Provide a structured process for identifying, containing, eradicating, and recovering from security incidents affecting Orchid Ranker deployments.

## Roles & Contacts
- **Incident Commander (IC):** Product Manager (or delegate)
- **Security Lead:** Security/Privacy engineer
- **DevOps Lead:** On-call infrastructure engineer
- **Communications Lead:** GTM lead / customer success

Contact roster to be maintained in shared on-call rotation document.

## Severity Levels
- **SEV-1 (Critical):** Data breach, active compromise of customer environments, or DP parameter leakage.
- **SEV-2 (High):** Major service degradation, unauthorized access attempts, or failed DP audits.
- **SEV-3 (Moderate):** Suspicious activity without confirmed compromise, minor availability issues.

## Response Phases
1. **Detection & Triage**
   - Monitor audit logs, SIEM alerts, customer reports.
   - IC assigns severity within 30 minutes of detection.
2. **Containment**
   - Disable affected credentials, isolate workloads, stop DP training jobs if required.
3. **Eradication & Recovery**
   - Patch vulnerabilities, rotate keys, redeploy clean artefacts.
   - Validate system integrity before restoring operations.
4. **Post-Incident Review**
   - Conduct blameless retro within 5 business days.
   - Document timeline, root cause, corrective actions.

## Communication Plan
- Internal war-room (Slack/Teams bridge) initiated by IC.
- Customer notification within 24 hours for SEV-1 incidents, summarizing impact and mitigation steps.
- Regulatory/legal notification coordinated with counsel when required.

## Artefact Handling
- Preserve relevant logs (audit, application, infrastructure) for forensic analysis.
- Snapshot affected systems where feasible.

## Testing & Drills
- Tabletop exercise at least annually.
- Update playbook based on lessons learned.

## Change Management
- Document corrective actions in issue tracker with owner and due date.
- Feed security learnings back into roadmap (e.g., additional hardening tasks).
