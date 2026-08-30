# signlred

`signlred` is not a generic `thiserror` replacement.

It is responsible for the **quality of machine-learning and linear-algebra results**:
whether a fit is identified, whether a matrix inverse is meaningful, whether an
online update carried information, and whether a statistically vacuous computation
was performed.

Every recoverable computation in `kanimiso` returns either:

- `Err(Failure)` when the result must not be used, or
- `Ok(Qualified<T>)` when a value exists together with a `Report` of warnings,
  numerical compromises, and meaninglessness diagnoses.

Callers who discard the `Report` are discarding the quality contract.
