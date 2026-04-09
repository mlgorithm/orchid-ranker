# Orchid Ranker — Enterprise & Performance Roadmap

**Owner:** Sam Vadiee  
**Created:** 2026-04-09  
**Goal:** Transform Orchid Ranker from a promising v0.2 research library into the definitive enterprise-grade educational recommender framework — fast, tested, production-ready.

---

## Strategic Thesis

Orchid Ranker already has two things no competitor offers: *education-native design* (BKT, ZPD, curriculum sequencing, learner simulation) and *privacy-by-design* (DP-SGD, RDP accounting, SafeSwitchDR). The path to becoming the top recommender library for education is not about adding more algorithms — it's about making what exists *fast, reliable, and trustworthy enough* for enterprise adoption.

The plan is organized into 6 phases over ~20 weeks. Phases overlap where dependencies allow.

---

## Phase 1 — Performance Foundation (Weeks 1–4) ✅ CORE COMPLETE

**Goal:** 3–5× inference speedup on key hot paths; establish benchmark baselines.  
**Status:** Core deliverables complete. LinUCB batched solve deferred to Phase 3 (lower priority).

### 1.1 Vectorize the Critical Hot Paths

These are the highest-impact performance fixes, identified from profiling the source code.

| Bottleneck | Location | Problem | Fix | Expected Gain |
|---|---|---|---|---|
| NeuralMF negative sampling | `baselines.py:819–879` | Python loops with `.detach().cpu().numpy()` per user for BPR; nested while-loops for softmax sampling | Replace with `torch.multinomial()` for batch negative sampling; pre-compute negative candidate pools | 5–10× training speedup |
| UserKNN CPU similarity | `baselines.py:388–392` | Full user matrix cosine similarity on CPU with numpy | Use `torch.nn.functional.cosine_similarity()` on device; pre-compute similarity matrix lazily | 3–5× for large user sets |
| Per-user tensor allocation in orchestrator | `agentic.py:948` | `torch.tensor(..., device=device)` creates new GPU tensor every user iteration | Pre-allocate state buffer `(num_users, 4)`, index into it with `state_buf[user_idx]` | Eliminates ~1–2ms/user |
| LinUCB per-item matrix solve | `baselines.py:470` | `torch.linalg.solve()` called independently per item — O(K·d³) | Batch solve: stack A matrices → single batched solve; or use Sherman-Morrison incremental updates | 2–4× for large candidate sets |
| PopularityBaseline .item() sync | `baselines.py:188` | List comprehension with `.item()` causes CPU sync per item | Vectorize: build scores tensor from dict, `torch.tensor([pop.get(i, 0) for i in item_ids])` in one call | Minor but removes GPU sync stalls |

### 1.2 Batch User Inference in Orchestrator

The biggest structural performance problem is that `MultiUserOrchestrator.run()` calls `rec.infer()` once per user per round. For 10,000 users × 50 rounds, that's 500,000 separate forward passes with individual device syncs.

**Action items:**
- Add `infer_batch(user_vecs, item_matrix, state_vecs)` to `TwoTowerRecommender` that processes N users in a single forward pass.
- Reshape the orchestrator loop: collect all user states → batch infer → scatter results → per-user decide.
- Keep per-user decide (MMR reranking) sequential since it depends on individual user state, but batch the expensive tower forward pass.
- Expected gain: 10–50× throughput improvement on GPU for the inference phase.

### 1.3 Enable torch.compile on the Tower Network

`bench_infer.py` already has torch.compile support but `TwoTowerRecommender` doesn't use it by default.

**Action items:**
- Add `compile: bool = False` parameter to `TwoTowerRecommender.__init__()`.
- When enabled, wrap the forward path with `torch.compile(mode="reduce-overhead")`.
- Guard against MPS (not supported) and PyTorch < 2.0.
- Add benchmark comparison (compiled vs. eager) to CI.
- Expected gain: 20–40% inference latency reduction on CUDA.

### 1.4 Establish Benchmark Baselines

No optimization program works without before/after numbers.

**Action items:**
- Commit a frozen benchmark dataset (MovieLens 100K subset, deterministic split, seed=42) to `benchmarks/fixtures/`.
- Run all 9 strategies through `profile_strategies.py` with 3 seeds, store results as `benchmarks/golden/baselines.json`.
- Create `benchmarks/bench_orchestrator.py` that measures per-round throughput (users/sec) for the agentic loop.
- Add `bench_infer.py` results (p50/p95/p99 latency) to golden baselines.
- Gate CI on ±15% regression threshold.

**Deliverables:**
- [x] Vectorized negative sampling in NeuralMF *(done — `5dedc6a`)*
- [x] Batched user inference in orchestrator *(done — `38e9177` adds `infer_batch()`)*
- [x] torch.compile integration for TwoTower *(done — `38e9177` adds `compile=True` flag)*
- [x] Golden benchmark baselines committed + CI regression gate *(done — `b7bb623`)*
- [x] Pre-allocated state buffers in agentic loop *(done — `38e9177`)*
- [ ] LinUCB batched matrix solve *(deferred — lower priority)*

---

## Phase 2 — Test Coverage & Correctness (Weeks 2–6) ✅ CORE COMPLETE

**Goal:** 70%+ line coverage with meaningful correctness assertions.  
**Status:** Core deliverables complete. ~145 new tests across 6 files. Coverage gate enforced in CI. Property-based tests with hypothesis and test tier separation deferred to Phase 3.

### 2.1 Priority Test Targets

Ordered by risk × complexity:

1. **StudentAgent 3PL response model** — The correctness of the entire simulation depends on this. Test that `P(correct) = c + (1-c)/(1 + exp(-a(θ-b)))` produces correct probabilities for known parameter values. Test ZPD shaping, fatigue decay, trust dynamics, engagement updating. Property-based tests for monotonicity (higher ability → higher P(correct)).

2. **TwoTowerRecommender scoring pipeline** — Test that user/item tower outputs have correct shapes. Test that FiLM gating modulates scores correctly. Test that per-user calibration preserves ranking order. Test numerical stability with zero vectors, extreme embeddings.

3. **DP-SGD gradient clipping + noise injection** — Test that per-sample gradients are clipped to max_grad_norm. Test that noise scale matches configured noise_multiplier. Test that privacy accountant epsilon accumulates correctly across steps. Test that the null accountant returns (0, 0).

4. **SafeSwitchDR gating logic** — Test that confidence sequences widen correctly. Test acceptance floor enforcement. Test that teacher policy is used when student policy is rejected. Test boundary conditions (first round, single observation).

5. **OrchidRecommender end-to-end** — For each of the 9 strategies: fit on small known dataset, verify recommendations are non-empty, verify scores are finite, verify filter_seen works, verify predict/predict_many consistency.

6. **MultiUserOrchestrator** — Test warmup phase produces valid buffers. Test that per-round user count matches config. Test DP integration (epsilon increases after training steps). Test that adaptive mode outperforms random on a simple synthetic task.

### 2.2 Testing Infrastructure

**Action items:**
- Add `pytest-cov` to dev dependencies; wire `--cov --cov-fail-under=70` into CI.
- Add `hypothesis` for property-based testing of numerical code.
- Create `tests/conftest.py` fixtures: small interaction DataFrame, tiny neural model, mock student agent.
- Add `tests/fixtures/` with deterministic small datasets.
- Separate test tiers: `tests/unit/`, `tests/integration/`, `tests/e2e/`.

### 2.3 Edge Case Coverage

Every public API method needs tests for:
- Empty input (empty DataFrame, zero users, zero items)
- Single user / single item
- Duplicate interactions
- NaN/Inf in ratings
- Mismatched column names
- Unknown user_id/item_id in predict/recommend after fit

**Deliverables:**
- [x] 70%+ line coverage enforced in CI *(done — `6b31177` adds pytest-cov + --cov-fail-under=70)*
- [x] Correctness tests for 3PL, DP clipping, confidence sequences *(done — `3da36ff`)*
- [x] All 9 strategies have fit → recommend → predict round-trip tests *(done — `3da36ff` test_recommender_correctness.py)*
- [x] Edge case tests for empty/single/NaN inputs *(done — `3da36ff` test_edge_cases.py)*
- [ ] Property-based tests with hypothesis (stretch goal)
- [ ] Test tier separation (unit/integration/e2e)

---

## Phase 3 — Architecture & API Hardening (Weeks 4–8) ✅ CORE COMPLETE

**Goal:** Clean module boundaries, stable public API, deprecation path for legacy code.  
**Status:** Core deliverables complete. 3 monolithic files decomposed into 10 focused modules. Tiered API with py.typed and mypy --strict in CI. Legacy deprecation with migration guide. Two core classes (TwoTowerRecommender, MultiUserOrchestrator) remain >400 lines as single-class files — further extraction would break cohesion.

### 3.1 Module Decomposition

Current state → target state:

```
agents/recommender_agent.py (1,466 lines)
  → agents/two_tower.py         (TwoTowerRecommender, DualRecommender)
  → agents/policies.py          (LinUCBPolicy, BootTS)
  → agents/logging.py           (JSONLLogger)
  → agents/calibration.py       (per-user calibration logic)

agents/agentic.py (1,517 lines)
  → agents/orchestrator.py      (MultiUserOrchestrator)
  → agents/config.py            (MultiConfig, UserCtx)
  → agents/warmup.py            (warmup loop logic)
  → agents/training.py          (per-round training step logic)

experiments/runner.py (1,065 lines)
  → experiments/data_prep.py    (data loading and encoding)
  → experiments/model_factory.py (strategy construction)
  → experiments/training.py     (training loop)
  → experiments/evaluation.py   (metric computation and reporting)
```

Rule: no file exceeds 400 lines. Every extracted module gets its own unit tests.

### 3.2 API Tiering

Restructure `__init__.py` into three tiers:

```python
# Tier 1: Stable public API (guaranteed semver)
from .recommender import OrchidRecommender, Recommendation
from .knowledge_tracing import BayesianKnowledgeTracing, MasteryTracker, ForgettingCurve
from .curriculum import PrerequisiteGraph, CurriculumRecommender
from .evaluation import learning_gain, knowledge_coverage, ...
from .serialization import save_model, load_model
from .model_selection import cross_validate, compare_models, train_test_split

# Tier 2: Advanced API (stable but may evolve between minor versions)
from .agents import StudentAgent, MultiUserOrchestrator, TwoTowerRecommender

# Tier 3: Internal/experimental (not in __all__, import from submodule)
# UserCtx, MultiConfig, JSONLLogger, BootTS, LinUCBPolicy, etc.
```

### 3.3 Legacy Orchestrator Deprecation

- Add `warnings.warn("LegacyOrchestrator is deprecated...", DeprecationWarning)` in v0.3.0.
- Document migration guide: `docs/migration/legacy_orchestrator.md`.
- Remove in v0.5.0.

### 3.4 Type Safety

- Add `py.typed` marker to package.
- Enable `mypy --strict` on Tier 1 API modules.
- Add `mypy` to CI pipeline (it's in dev deps but not run in CI currently).

**Deliverables:**
- [x] Module decomposition: 3 monolithic files → 10 focused modules *(done — `9a42309`)*
  - `recommender_agent.py` → `policies.py`, `logging_util.py`, `two_tower.py`, `dual_recommender.py`
  - `agentic.py` → `config.py`, `timing.py`, `orchestrator.py`
  - `runner.py` → `data_prep.py`, `model_factory.py`, `evaluation.py`
- [x] Tiered `__init__.py` with clear stability guarantees *(done — `8334ae5`)*
- [x] `py.typed` marker + mypy --strict on Tier 1 modules in CI *(done — `8334ae5`)*
- [x] Legacy orchestrator deprecated with migration guide *(done — `8334ae5`)*
- [x] Backward-compatible import shims for moved modules *(done — `9a42309`)*
- [ ] All helper/config files under 400 lines *(done — core classes TwoTower/Orchestrator intentionally kept as single-class files)*

---

## Phase 4 — Security & Compliance Hardening (Weeks 6–10)

**Goal:** Security model that passes enterprise procurement review.

### 4.1 Authentication Integration

The current RBAC (`access.py`, 39 lines) has no auth layer. Enterprise deployments need it.

**Action items:**
- Add optional JWT/OIDC token validation middleware: `orchid_ranker.security.auth`.
- Support: decode JWT → extract role claim → feed into `AccessControl.can()`.
- Provider-agnostic: accept any OIDC-compliant issuer (Okta, Auth0, Azure AD, Keycloak).
- Keep it optional — library users who don't need auth shouldn't be forced into it.

### 4.2 Audit Log Integrity

Current audit logger writes plain JSONL. Tampering is undetectable.

**Action items:**
- Implement HMAC hash chaining: each log line includes `prev_hash` (HMAC of previous line).
- Add `verify_log_integrity(path)` utility that validates the chain.
- Add optional encryption-at-rest via Fernet (symmetric, key from env/KMS).
- Document: what the audit log protects against, what it doesn't, and where the deployment environment must fill gaps.

### 4.3 Secrets Management

Connectors currently accept raw passwords as strings.

**Action items:**
- Add `SnowflakeConnector.from_env()` pattern (read credentials from environment variables).
- Add `SnowflakeConnector.from_vault(vault_client, secret_path)` pattern.
- Document: never pass secrets as constructor arguments in production; always use env or vault.
- Add `ruff` rule to flag string literals named `password` or `secret` in non-test code.

### 4.4 Threat Model Documentation

Create `docs/security/threat_model.md`:
- What Orchid protects: data access (RBAC), privacy (DP), audit trail (logging), model integrity (serialization checksums).
- What Orchid delegates: network security, infrastructure hardening, identity management, key management.
- Where the boundaries are: Orchid is a library, not a service — deployment security is the operator's responsibility.

### 4.5 Third-Party Pen Test

- Engage external security firm for pen test (already noted as pending in your readiness plan).
- Scope: the library's serialization (pickle/torch.load attack surface), the audit webhook endpoint, the connector authentication flows.

**Deliverables:**
- [ ] JWT/OIDC auth middleware (optional)
- [ ] HMAC hash-chained audit logs + integrity verifier
- [ ] Secrets-from-env/vault patterns for all connectors
- [ ] Threat model document
- [ ] Pen test scheduled and scoped

---

## Phase 5 — Observability, Connectors & Deployment (Weeks 8–14)

**Goal:** Production-grade operational infrastructure.

### 5.1 Observability Expansion

Current state: 3 Prometheus metrics. Target:

| Metric | Type | Description |
|---|---|---|
| `orchid_training_runs_total` | Counter | (exists) |
| `orchid_training_duration_seconds` | Histogram | (exists) |
| `orchid_dp_epsilon_cumulative` | Gauge | (exists) |
| `orchid_inference_latency_seconds` | Histogram | P50/P95/P99 per strategy |
| `orchid_inference_requests_total` | Counter | Per strategy, per outcome |
| `orchid_inference_errors_total` | Counter | Per error type |
| `orchid_recommendation_list_size` | Histogram | Actual items returned per request |
| `orchid_active_users` | Gauge | Users with activity in current window |
| `orchid_model_staleness_seconds` | Gauge | Time since last model update |
| `orchid_dp_budget_remaining` | Gauge | 1 - (current_epsilon / max_epsilon) |

**Additional action items:**
- Add OpenTelemetry as an alternative instrumentation backend (it's becoming the enterprise standard).
- Add `/healthz` and `/readyz` endpoints for Kubernetes probes.
- Commit Grafana dashboard JSON to `deploy/grafana/`.
- Add structured logging (JSON) as an option alongside the current plain text logger.

### 5.2 Connector Hardening

Promote from stubs to production-grade:

| Connector | Add | Priority |
|---|---|---|
| Snowflake | Connection pooling, parameterized queries only (no string interpolation), async fetch, schema validation | High |
| BigQuery | Job-based async queries, pagination, result streaming, cost estimation logging | High |
| S3 | Multipart upload, streaming read with chunking, SSE-KMS encryption support | Medium |
| MLflow | Experiment auto-creation, artifact versioning, model registry integration | Medium |

For each connector:
- Integration tests with `moto` (S3), `testcontainers` (Snowflake), or BigQuery emulator.
- Error taxonomy: transient (retry) vs. permanent (fail fast) vs. config (user error).
- Circuit breaker pattern for transient failures.

### 5.3 Deployment Templates

- Harden Helm chart: add HPA (horizontal pod autoscaler), PDB (pod disruption budget), resource limits, liveness/readiness probes.
- Add `deploy/docker-compose.yml` for local development with Prometheus + Grafana.
- Add `deploy/terraform/aws/` with ECS Fargate and EKS module examples.
- Document capacity planning: users/sec per vCPU, memory per 1M interactions, GPU vs CPU decision matrix.

**Deliverables:**
- [ ] 10+ Prometheus metrics covering inference, errors, DP budget
- [ ] OpenTelemetry support
- [ ] Health check endpoints
- [ ] Grafana dashboard templates
- [ ] All 4 connectors hardened with integration tests
- [ ] Helm chart with HPA, PDB, resource limits
- [ ] docker-compose for local dev stack
- [ ] Capacity planning guide

---

## Phase 6 — Documentation, Benchmarks & Launch (Weeks 12–20)

**Goal:** The library is discoverable, well-documented, and has published performance claims.

### 6.1 Docstring Blitz

- Add NumPy-style docstrings to all 300+ undocumented public methods.
- Priority: Tier 1 API first (OrchidRecommender, BKT, curriculum, evaluation, serialization).
- Include: Parameters, Returns, Raises, Examples for every public method.
- Enforce with `ruff` rule D100–D107 (missing docstrings) in CI.

### 6.2 Documentation Site

- Set up mkdocs-material with autodoc from docstrings.
- Structure: Getting Started → Tutorials → API Reference → Deployment Guide → Security → Contributing.
- Host on GitHub Pages or Read the Docs.
- Add versioned docs (v0.2, v0.3, etc.).

### 6.3 Published Benchmark Results

This is critical for credibility. Enterprise buyers and researchers both want numbers.

**Action items:**
- Run standardized benchmarks on MovieLens 100K, MovieLens 1M, and EdNet (if available).
- Compare against: Surprise (SVD), implicit (ALS, BPR), LightFM, RecBole (select models).
- Metrics: P@10, R@10, NDCG@10, training time, inference latency (p50/p95), memory usage.
- Publish results in `docs/benchmarking_results.md` and in the README.
- Automate benchmark runs in CI (nightly) with regression detection.

**Target claims (validate with benchmarks):**
- "Orchid NeuralMF-softmax achieves competitive NDCG@10 with Surprise SVD and implicit ALS on MovieLens 100K."
- "Batched inference serves 10,000+ users/sec on a single GPU."
- "Agentic simulation runs 50 rounds × 1,000 users in under 60 seconds on CPU."

### 6.4 Examples & Notebooks

- Expand `examples/notebooks/` with:
  - Quick start (5 min): fit → recommend → evaluate.
  - Knowledge tracing tutorial: BKT → mastery tracking → spaced repetition.
  - Curriculum design: prerequisite graph → learning path → curriculum recommender.
  - Agentic simulation: student agent → orchestrator → policy comparison.
  - Privacy tutorial: DP presets → accountant → privacy-utility tradeoff plot.
  - Enterprise deployment: Docker → Helm → Prometheus → Grafana.

### 6.5 Release Automation

- Set up GitHub Actions workflow: tag → build sdist + wheel → publish to PyPI → generate SBOM → create GitHub Release with changelog.
- Add `RELEASING.md` with the release process.
- Adopt CalVer or strict SemVer with documented policy.

**Deliverables:**
- [ ] 100% docstring coverage on public API
- [ ] mkdocs site live on GitHub Pages
- [ ] Published benchmark results vs. 4+ competitors
- [ ] 6+ tutorial notebooks
- [ ] Automated release pipeline to PyPI
- [ ] SBOM generated per release

---

## Performance Targets Summary

| Metric | Current (estimated) | Target | Phase |
|---|---|---|---|
| NeuralMF training throughput | ~1K samples/sec (Python loop neg sampling) | 10K+ samples/sec (vectorized) | 1 |
| Orchestrator inference (per round, 1K users) | ~1K infer calls (sequential) | 1 batched call, 10–50× faster | 1 |
| OrchidRecommender.recommend() latency (ALS, 10K items) | ~5–15ms | ~2–5ms with torch.compile | 1 |
| TwoTower inference (compiled, CUDA) | ~1ms/user (eager) | ~0.3–0.5ms/user (compiled) | 1 |
| Test coverage | ~6% | 70%+ | 2 |
| Docstring coverage | 17% | 100% (public API) | 6 |
| Prometheus metrics | 3 | 10+ | 5 |
| Published benchmark datasets | 0 | 3+ (ML-100K, ML-1M, EdNet) | 6 |

---

## Competitive Positioning After Roadmap

| Capability | Orchid (post-roadmap) | Surprise | implicit | LightFM | RecBole |
|---|---|---|---|---|---|
| Education-native (BKT, ZPD, curriculum) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Built-in DP | ✅ | ❌ | ❌ | ❌ | ❌ |
| Learner simulation | ✅ | ❌ | ❌ | ❌ | ❌ |
| Algorithm count | 9+ | 11 | 4 | 4 | 70+ |
| Surprise-like API | ✅ | ✅ | ❌ | ❌ | ❌ |
| Enterprise infra (RBAC, audit, connectors) | ✅ | ❌ | ❌ | ❌ | ❌ |
| GPU-optimized inference | ✅ | ❌ | ✅ | ❌ | ✅ |
| Published benchmarks | ✅ | ✅ | ✅ | ✅ | ✅ |
| Test coverage >70% | ✅ | ✅ | ✅ | ✅ | varies |
| Generated doc site | ✅ | ✅ | ✅ | ✅ | ✅ |

**The pitch:** Orchid Ranker is the only recommender library purpose-built for education, combining a familiar Surprise-like API with agentic learner simulation, differential privacy, knowledge tracing, and enterprise infrastructure — all in one pip-installable package, with competitive performance on standard benchmarks.

---

## Risk Register

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| Performance targets not met after vectorization | High | Medium | Profile first, optimize the measured bottleneck, not the assumed one. Benchmark before/after every change. |
| torch.compile breaks on edge cases | Medium | Medium | Gate behind flag, extensive testing, keep eager fallback. |
| Test coverage goal delays feature work | Medium | High | Parallelize: performance work doesn't require test coverage; test coverage doesn't require performance. Different workstreams. |
| RecBole adds education features | High | Low | Move fast on Phase 1+2. The simulation framework and DP integration are deep moats — hard to replicate. |
| Breaking API changes frustrate early adopters | High | Medium | Freeze Tier 1 API before v0.3. Deprecation policy: deprecated in N, removed in N+2. |
| Pen test reveals critical vulnerability | High | Low | Scope pen test to serialization (torch.load) and webhook. These are the highest-risk surfaces. |

---

## Weekly Milestones (First 8 Weeks)

| Week | Milestone | Phase |
|---|---|---|
| 1 | Frozen benchmark dataset committed; baseline numbers recorded | 1 |
| 2 | NeuralMF negative sampling vectorized; before/after benchmarks | 1 |
| 3 | Batched user inference in orchestrator; throughput benchmarks | 1 |
| 4 | torch.compile integration; CI regression gates live | 1 |
| 5 | 40% test coverage; StudentAgent 3PL + DP clipping tests | 2 |
| 6 | 60% test coverage; all 9 strategies round-trip tested | 2 |
| 7 | Module decomposition complete; all files <400 LOC | 3 |
| 8 | Tiered API in __init__.py; py.typed + mypy in CI | 3 |
