# denshi

`denshi` is a small, verification-first crate for online prediction and
full-information regret minimization. It currently provides:

- `Hedge`, the exponentially weighted expert forecaster for losses in `[0, 1]`;
- projected `OnlineGradientDescent` for caller-supplied convex-loss gradients.

The API makes the online protocol explicit: read the decision, incur its loss,
and only then call `update`. Algorithms reject non-finite feedback instead of
silently repairing it. There are no runtime dependencies and no hidden RNG.

```rust
use denshi::Hedge;
let mut learner = Hedge::new(2, 0.5)?;
let mixture = learner.probabilities();
let loss = learner.update(&[0.0, 1.0])?;
assert_eq!(mixture, [0.5, 0.5]);
assert_eq!(loss, 0.5);
# Ok::<(), denshi::Error>(())
```

See [`docs/validation.md`](docs/validation.md) for scope and evidence.
