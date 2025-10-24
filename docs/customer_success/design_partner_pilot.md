# Design Partner Pilot Plan (Draft)

## Goals
- Validate Orchid Ranker in real-world pipelines while keeping the project library-first.
- Capture testimonials and quantitative benchmarks to inform roadmap and documentation.

## Pilot Timeline (6 Weeks)
1. **Week 0 (Planning):** Finalise scope, datasets, target metrics, and success criteria. Confirm privacy/DP parameters.
2. **Week 1 (Environment):** Customer installs package via pip, runs smoke tests, and configures audit/metrics endpoints.
3. **Week 2-3 (Integration):** Implement ingestion/preprocessing, wire evaluation scripts, iterate on recommenders.
4. **Week 4 (Experimentation):** Run adaptive vs baseline experiments, log metrics via Prometheus/MLflow connectors.
5. **Week 5 (Review):** Joint analysis session; capture insights, blockers, and potential feature requests.
6. **Week 6 (Wrap-up):** Gather testimonial draft (template below), publish findings (with customer approval), and plan next iteration.

## Deliverables
- Benchmark summary (Precision@K, NDCG, training cost).
- Written testimonial or quote focusing on library integration and results.
- Notebook or script contributed back to `examples/notebooks/` if NDA permits.

## Testimonial Template
> **Context:** (Describe problem/data size)
>
> **Outcome:** (Quantified improvement, e.g., “Improved Precision@5 by 12%”) 
>
> **Experience:** (Feedback on installation, APIs, docs)
>
> **Quote:** (One-sentence endorsement, optional branding approval)

## Feedback Loop
- Record feature requests in GitHub with `design-partner` label.
- Update onboarding/support playbooks with new learnings.
- Share monthly progress recap with community (blog/newsletter).
