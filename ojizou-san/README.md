# ojizou-san

`ojizou-san` is not `log` / `tracing`.

It is responsible for a durable, inspectable **quality ledger** of machine-learning
and linear-algebra work:

- numerical compromises (what was intended vs what was actually computed)
- meaningless-fit alerts
- incremental / online update explainability
- optimization and convergence traces

`kanimiso` algorithms must write to an `ojizou-san` session. The ledger is part of
the result, not an optional debug print.
