# pyKT Integration

pyKT is a research benchmark library for deep knowledge tracing. Orchid should
not try to replace that model zoo. The practical integration pattern is:

1. Use Orchid to normalize adaptive-learning logs.
2. Export sequences into pyKT's six-line learner format.
3. Train or evaluate a pyKT model in the research stack.
4. Bring predicted correctness probabilities back into Orchid for policy
   ranking, OPE, safety, audit, and deployment.

## Export Orchid Data To pyKT

pyKT's raw sequence format stores each learner in six lines:

1. learner id and sequence length
2. question ids
3. concept ids
4. binary responses
5. timestamps
6. answering durations

```python
from orchid_ranker import export_pykt_sequences

export_pykt_sequences(
    interactions,
    "data/pykt/orchid_sequences.txt",
    user_col="user_id",
    item_col="item_id",
    correct_col="correct",
    concept_col="skill_id",
    timestamp_col="timestamp",
    duration_col="elapsed_time",
)
```

If concepts, timestamps, or durations are unavailable, Orchid writes `NA` for
that line's values, matching pyKT's documented raw format.

## Use pyKT Predictions In Orchid

After training a pyKT model, export a prediction table with one row per
user-item probability:

```text
user_id,item_id,p_correct
u1,q1,0.82
u1,q2,0.61
u1,q3,0.74
```

Then use the adapter anywhere Orchid expects a KT-style predictor:

```python
from orchid_ranker.learning_policy import KTValuePolicy
from orchid_ranker.pykt_bridge import PyKTPredictionAdapter

tracer = PyKTPredictionAdapter.from_csv("pykt_predictions.csv")
policy = KTValuePolicy(tracer, target_correct=0.70)
ranked = policy.rank("u1", ["q1", "q2", "q3"], top_k=3)
```

`PyKTPredictionAdapter` is static: it does not update a pyKT model online. That
is deliberate. Use it to evaluate and serve exported model probabilities behind
Orchid's policy, OPE, and safety layers. For live learning, retrain/export the
pyKT model on your chosen cadence or use Orchid's native online tracers.

## Round-Trip Import

```python
from orchid_ranker.pykt_bridge import load_pykt_sequences, pykt_sequences_to_interactions

sequences = load_pykt_sequences("data/pykt/orchid_sequences.txt")
interactions = pykt_sequences_to_interactions(sequences)
```

This is useful for smoke tests, data audits, and verifying that exported
sequence files preserve learner order and response labels.
