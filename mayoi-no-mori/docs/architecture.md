# Architecture and ownership

The dependency direction is intentionally one-way:

```text
amatsuki (bits and sampling) ─┐
                              ├─> mayoi-no-mori (ensemble policy)
oldwood (CART gain + arenas) ─┘
```

`oldwood` owns matrix validation for CART, weighted impurity, exhaustive gain
evaluation, stable candidate ordering, deterministic tie-breaking, node arenas,
tree traversal, and feature importance. Its `SplitStrategy` hook allows an
ensemble to propose a node-local feature set and thresholds, but the hook cannot
score a split or construct nodes.

`mayoi-no-mori` owns:

- bootstrap and without-replacement row sampling;
- deterministic node-local feature proposals;
- ExtraTrees random threshold proposals;
- vote/probability/value aggregation and out-of-bag accounting;
- first-order residual objectives;
- discrete SAMME and AdaBoost.R2 sample reweighting;
- histogram construction, regularized Newton gains/leaf values, and its arena;
- ordered target statistics for categorical columns.

Histogram boosting also uses a distinct second-order kernel: aggregate
gradient/Hessian scores are not weighted-SSE CART on per-row pseudo-targets.
It shares oldwood's `NodeKind`, options and proposal contract, but evaluates
`soft_threshold(G,L1)^2/(H+L2)` via compensated, scaled bin reductions. Prefix
and suffix scans avoid total-minus-prefix cancellation. Min-gain is raw score
gain without a half factor; min-weight is interpreted as minimum leaf Hessian.

Isolation Forest is another deliberate exception to the ordinary CART boundary:
its random partitions optimize no impurity and therefore use a separate,
compact arena rather than pretending to be CART.

This boundary is why a forest, first-order boosting stage, and ExtraTrees model expose
the same inspected `oldwood` arena instead of carrying similar private node
types.

## Determinism

A root ChaCha8 stream produces row samples and per-tree seeds. A split-strategy
stream is derived from `(seed, node_id, depth, sample_count, feature)`, so a
callback's output does not depend on the order in which other callbacks were
invoked. `oldwood` then sorts and deduplicates candidates before scoring.

## Upstream-name policy

`LightGbm*` and `CatBoost*` are algorithm-family entry points, not compatibility
claims. Each type's rustdoc and the README enumerate the subset. Adding an
upstream capability requires a new differential fixture or closed-form test;
renaming an existing algorithm is not accepted as support.
