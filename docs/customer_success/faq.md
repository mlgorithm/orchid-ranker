# Customer FAQ (Draft)

## Installation
> **Q:** How do I install the library?

`pip install orchid-ranker` (with optional extras: `[agentic]`, `[viz]`, `[observability]`, `[connectors]`).

## Dataset Requirements
> **Q:** What schema do datasets need?

Training datasets require at minimum `user_id`, `item_id`, and an implicit/explicit label column. Side info optional. See `docs/overview.md` and `docs/deployment.md`.

## Differential Privacy
> **Q:** Does Orchid Ranker support DP training?

Yes – configure `dp_cfg` with engine `per_sample` or `opacus`. Audit logs record epsilon deltas automatically.

## Observability
> **Q:** How do I monitor model training?

Use `orchid_ranker.start_metrics_server()` for Prometheus metrics and `AuditLogger` for DP audit trails. Deployment quickstart covers log forwarding.

## Support
> **Q:** Where do I get help?

Open a GitHub issue or, for design partners, use the dedicated Slack channel. See `support_sla.md` for response targets.

## Roadmap Requests
> **Q:** How can I suggest new features?

Submit an issue labeled `enhancement` with context, desired outcome, and sample data if applicable.
