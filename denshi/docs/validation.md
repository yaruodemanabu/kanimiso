# Validation

The Hedge replay checks its softmax mixture against the two-expert closed form,
normalization, bounded probabilities, cumulative losses, and behavior after a
large loss separation. The projected-gradient replay checks the analytic
projection of `(3, 4)` onto the unit Euclidean ball and feasibility after a
second step. Error tests verify that malformed feedback does not advance the
round counter.

The current scope deliberately excludes contextual features, delayed feedback,
and stochastic bandit feedback; the latter belongs to `riko`.

## Design references

The implementation follows the online-learning protocol and regret viewpoint
in the [requested Kodansha reference](https://www.kspub.co.jp/book/detail/1529229.html).
The public decision-before-feedback split was also compared with the learner
interfaces in [Vowpal Wabbit](https://github.com/VowpalWabbit/vowpal_wabbit).
Neither source is treated as a numerical oracle: the committed tests replay
the defining equations directly.
