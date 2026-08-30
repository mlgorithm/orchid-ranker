# Design-partner council

Orchid uses a small customer council to pressure-test product decisions before
they become roadmap commitments. These are **simulated design-partner roles**
for product discovery and review. They do not represent, speak for, or imply a
relationship with any named company.

## The council

| Role | Job to be done | Veto questions |
| --- | --- | --- |
| Learning-program owner | Improve retained mastery without giving up the authored curriculum | Is the primary outcome independent and delayed? Is there a static control? |
| Learning designer/instructor | Keep pedagogical control and understand every adaptation | Can required units be locked, recommendations explained, and the authored path restored? |
| Platform engineer | Embed Orchid safely in an existing learning product | Are decisions/outcomes durable and idempotent? Can any learner path be replayed? |
| Learner advocate | Protect the learner experience | Does adaptation avoid repeated failure, excessive repetition, inaccessible content, or unexplained difficulty jumps? |
| Media-learning observer | Test future audio/video-learning extensions without diluting the product | Is this assessed, structured learning—or merely engagement optimization? |

The last role can be informed by a hypothetical Spotify- or YouTube-like
platform. It must not be used to imply that those companies are customers or
partners.

## Review every proposed feature

Before accepting a feature, each council role answers these questions:

1. **Learning outcome:** What delayed, independent mastery outcome could this
   improve? Immediate correctness and watch time are not enough.
2. **Curriculum control:** Which eligibility, prerequisite, assessment-holdout,
   and override rules remain outside the model?
3. **Data contract:** What exact event, catalog version, candidate set, decision
   ID, and outcome join are required?
4. **Failure mode:** How does the learner fall back to an authored path if data
   is sparse, content is new, or the service is unavailable?
5. **Evidence:** What static control and randomized experiment would make the
   result believable?

Reject features that cannot answer all five. Put general engagement ranking,
unbounded content retrieval, and advanced policy learning in a separate future
track unless they meet the same learning-evidence bar.

## Current council decisions

### Target customer

Start with a professional-certification, technical-skills, test-preparation, or
other assessed-practice product. It should have a stable exercise bank, a skill
map, several valid next exercises, automated scoring, and a delayed assessment.

### Not the first market

Do not target a home feed, music radio, podcast discovery, or general video
recommendation. Those systems optimize different objectives and need retrieval,
implicit-feedback debiasing, multi-objective ranking, and real-time experiment
infrastructure beyond Orchid's adaptive-practice scope.

Audio/video learning is in scope only when it is a bounded course with scored
practice or checkpoints. A video view or audio completion is context, not proof
of mastery.

### Delivery gates

The council will not approve a production pilot until Orchid has:

1. A versioned curriculum catalog with course/module, assessment-holdout,
   skill, difficulty, and prerequisite diagnostics.
2. Durable, idempotent decision/outcome storage; deterministic replay; and a
   visible authored-path fallback/rollback.
3. A reference event/service integration and recommendation explanations for
   learning designers.
4. A pre-registered learner-level experiment against the current authored path,
   with delayed independent mastery as the primary outcome.

These delivery gates refine the [product roadmap](roadmap.md); they do not
replace the existing [data-readiness](guides/00-adaptive-practice.md) or
[learning-pilot](guides/03-learning-pilot.md) requirements.
