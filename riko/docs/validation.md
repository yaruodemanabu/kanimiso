# Validation

The UCB replay verifies forced initialization of every arm and the analytic
post-initialization choice. The exponential-weights replay compares one update
with its closed form at zero tolerance. Property tests cover normalized
probabilities, inverse-CDF boundaries, bounded rewards, and stale-feedback
rejection without state advancement.

The current verified core deliberately excludes contextual, combinatorial,
continuum-armed, and non-stationary bandits, and posterior-sampling policies.

## Design references

The stochastic/adversarial distinction and interaction protocol follow the
[requested Kodansha reference](https://www.kspub.co.jp/book/detail/1529175.html).
Policy structure was compared with
[SMPyBandits](https://github.com/SMPyBandits/SMPyBandits) and
[Vowpal Wabbit](https://github.com/VowpalWabbit/vowpal_wabbit), while retaining
a dependency-free Rust implementation. Those projects are design references,
not runtime dependencies or numerical oracles; the tests use closed forms.
