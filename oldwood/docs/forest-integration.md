# Reusing CART from a forest crate

`oldwood` owns matrix validation, weighted statistics, candidate scoring,
tie-breaking, and arena construction. A forest crate should own only sampling,
random-number generation, aggregation, and forest diagnostics.

## Bootstrap rows

A bootstrap sample is a sequence of row indices and can contain duplicates.
Expose that sequence through a small `MatrixView` adapter and construct target
and weight views in the same order. This preserves bootstrap multiplicity for
`min_samples_split` and `min_samples_leaf` without copying feature storage.

Encoding bootstrap multiplicity only by multiplying weights is not equivalent
when row-count stopping rules are active.

## Node-specific feature and threshold policies

Implement `SplitStrategy`. `SplitContext` contains stable `node_id`, `depth`,
and positive-weight `sample_count`; a caller can derive a deterministic local
seed from `(forest_seed, tree_id, node_id, depth)`.

`features` may return a node-specific feature subset. `thresholds` receives
the stable sorted unique values for one feature and may return exhaustive or
random thresholds. The engine then:

- rejects out-of-range features;
- rejects non-finite thresholds or thresholds outside `[minimum, maximum)`;
- stably sorts and deduplicates candidates;
- enforces `TreeOptions::max_features`;
- evaluates impurity and child constraints;
- resolves equal gain by `(feature, threshold)`.

Consequently a forest or extremely randomized tree never needs to reproduce a
criterion, weighted moment, split scan, or node builder.

## Deliberately absent from oldwood

- RNG algorithms and seed streams;
- bootstrap and subsampling policy;
- parallel scheduling;
- tree aggregation and class alignment across trees;
- out-of-bag accounting;
- forest-level feature importance and diagnostics.

Keeping those concerns outside this crate lets deterministic CART remain a
small numerical oracle for the randomized layer.
