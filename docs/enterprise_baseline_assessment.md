# Enterprise Baseline Assessment (Phase 0)

Date: 2025-02-15  
Owner: Product Management (Farhad Vadiee) with Engineering Leads

## 1. Architecture & Codebase Review

**Current State**
- Monorepo-style Python package under `src/orchid_ranker`, structured around recommender baselines, adaptive agentic simulator, and CLI tooling.
- Adaptive stack (`agents/`) depends on PyTorch 2.x features; agentic experiments run entirely in-process with JSONL logging.
- Packaging via `pyproject.toml` with extras for viz, agentic experiments, and benchmarks; no separate microservices or deployment manifests yet.

**Identified Gaps**
- No consolidated architecture diagram describing data flow (ingest → offline experiments → live integration).
- Deployment guidance is limited; lacks containerization/Helm/Terraform references.
- Differential privacy integration relies on custom implementation without external validation notes.

**Next Actions**
1. Produce a high-level architecture diagram (context + container views).
2. Draft deployment blueprint (batch + real-time scenarios).
3. Document DP threat model and limitations ahead of Phase 2.

## 2. Testing & Quality Coverage

**Current State**
- Pytest suite covers recommender API paths, CLI evaluation, DP helpers, metrics, and the agentic smoke run.
- New performance profiling harness (`benchmarks/profile_strategies.py`) exists, but regression thresholds are manual.
- Legacy compatibility tests (from `experiments/test_student.py`) now run against the modern `StudentAgent`.

**Identified Gaps**
- No GPU CI or matrix build presently documented (Phase 1 still to wire into automation).
- Lack of integration tests covering dataset ingestion and experiment orchestration end-to-end with real datasets.
- Performance guardrails rely on manual JSON comparisons; no automated check script committed.

**Next Actions**
1. Design CI matrix (CPU/GPU, Python versions) and document requirements.
2. Add end-to-end dataset fixture test once synthetic dataset bundle is prepared.
3. Commit regression checker for profiling outputs and integrate into CI spec.

## 3. Documentation Inventory

**Current State**
- README provides quickstart, strategy list, and support-matrix table.
- `docs/` directory hosts overview, benchmarking, privacy notes, and now enterprise readiness plan plus baseline assessment.
- No single-source user manual or API reference beyond README snippets.

**Identified Gaps**
- Missing migration guide for semantic versioning (should accompany Phase 1).
- No deployment playbook or troubleshooting guide.
- Privacy documentation lacks external references or compliance alignment.

**Next Actions**
1. Create docs index (user guide vs admin vs developer).
2. Draft initial deployment playbook skeleton.
3. Plan privacy/compliance whitepaper structure with legal/compliance input.

## 4. Priority Use Cases & SLAs

**Targeted Use Cases**
1. Offline benchmarking vs baseline recommenders (ALS, LinUCB, neural MF).
2. Agentic simulation for adaptive policy tuning with optional differential privacy.
3. Plug-in recommender API for existing EdTech platforms (batch scoring + online bandits).

**Initial SLA Targets**
- Offline experiment completion: < 6 hours per dataset (baseline), < 5% failure rate.
- Online inference SLA (when deployed): P95 latency < 150ms for top-K recommendation on CPU.
- DP reporting latency: privacy accountant summary available < 5 minutes post-run.

**Open Items**
- Validate SLA targets with design partners.
- Model resource requirements (GPU vs CPU) per use case.

## 5. Stakeholder Alignment

**Core Team**
- Product: Farhad Vadiee (PM)
- ML Engineering: Orchid core contributors (lead TBD)
- Infrastructure/DevOps: To be assigned (current gap)
- Privacy/Compliance: External advisor needed
- GTM/Customer Success: Not yet staffed

**Actions**
- Identify engineering and infra leads formally.
- Engage privacy consultant for DP review ahead of Phase 2.
- Outline GTM role requirements for future hiring.

## 6. Execution Cadence & Communication

**Proposed Cadence**
- Weekly tiger-team stand-up (30 mins) to track Phase goals.
- Bi-weekly roadmap review with broader stakeholders (engineering + product + GTM).
- Monthly exec summary / public roadmap update (GitHub Discussions or newsletter).

**Tooling**
- Use shared roadmap board (Notion/Jira equivalent) to track Phase deliverables.
- Centralize documentation in `docs/` with CHANGELOG-style updates per phase.

## 7. Risk & Dependency Log

| Risk | Description | Mitigation |
|------|-------------|------------|
| Privacy validation | Custom DP implementation may not pass enterprise audits | Fast-track Opacus integration in Phase 2; schedule external review |
| Deployment guidance | Lack of infra patterns could slow pilots | Prioritize deployment blueprint and containerization in Phase 3 |
| Resource availability | No dedicated infra/privacy owner | Identify internal champions or engage contractors ASAP |

## Summary

Phase 0 baseline assessment is documented. Phase 1 hardening tasks are underway/completed, with Phase 2 (Security & Privacy Foundations) next. This document should be revisited at each phase boundary to update risks, stakeholders, and priorities.
