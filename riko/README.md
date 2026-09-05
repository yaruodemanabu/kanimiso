# riko

`riko` is a small, verification-first multi-armed-bandit crate. It provides:

- `Ucb`, an upper-confidence-bound policy for stochastic rewards;
- `ExpWeights`, an importance-weighted exponential policy for adversarial rewards.

Rewards must be finite and in `[0, 1]`. Selection returns a `Choice` carrying
the arm, probability, and interaction round. Updates reject stale choices.
`ExpWeights` receives a uniform variate from the caller, so the crate has no
hidden RNG, seed, or external dependency.

```rust
use riko::Ucb;
let mut policy = Ucb::new(2, 2.0_f64.sqrt())?;
let choice = policy.select();
policy.update(choice, 1.0)?;
# Ok::<(), riko::Error>(())
```

See [`docs/validation.md`](docs/validation.md) for scope and evidence.
