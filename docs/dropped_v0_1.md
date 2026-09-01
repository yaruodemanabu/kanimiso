# v0.1 generated names dropped in 0.2

PR 8 deleted `kanimiso/src/hmm/legacy.rs` (568,846 lines). The generated
HMM families are not coming back. Use the v0.2 core instead.

## Kept (v0.2)

| Name | Replaces |
|---|---|
| `hmm::HiddenMarkovModel<E>` | every `*Hmm` wrapper + `Fitted*` pair |
| `hmm::Gaussian` | `GaussianHmm` / `GaussianHmmFull` / `Spherical` / `Tied` (diagonal only) |
| `hmm::Poisson` | `PoissonHmm` |
| `hmm::Categorical` | `MultinomialHmm` / `CategoricalHmm` |
| `hmm::CosinePower` | `Cosine3..173Hmm` (power is a runtime `f64`) |
| `hmm::TwoSidedPower` | `Tsp3..177Hmm` |
| `hmm::Transformed<E>` | `Unit*` / `Beta*` / `Kumaraswamy*` / `Exponentiated*` / `Discrete*` of a base law |

## Dropped

Every other `kanimiso::hmm::*` type that existed at `generated-v0.1-archive`
(`99c46d0`), including:

- Cosine-power / two-sided-power type-baked families (3,460 types)
- Variational / GMM / left-right aliases
- The long list of named emission HMMs (Gamma, Weibull, Student-t, …) that
  did not meet AGENTS.md §5 (oracle + one-file family + a written use)

Re-adding a dropped family requires an issue that names the oracle and a
single `Emission` implementation — not a new numbered type.
