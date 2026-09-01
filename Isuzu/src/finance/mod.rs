//! Financial engineering layer (Shreve I / II).
//!
//! Process generation stays in `models` / `simulate`. This module owns
//! payoffs, measures, analytic prices, Monte Carlo / tree / PDE engines,
//! optimal stopping, rate and jump products, and real options.

pub mod black_scholes;
pub mod brownian;
pub mod greeks;
pub mod hedging;
pub mod jumps;
pub mod market;
pub mod monte_carlo;
pub mod payoff;
pub mod pde;
pub mod rates;
pub mod realoption;
pub mod special;
pub mod stopping;
pub mod tree;

pub use black_scholes::{
    call as bs_call, digital_call, geometric_asian_call, margrabe, put as bs_put,
    put_call_parity_gap, BlackScholesMarket, BlackScholesPrice,
};
pub use brownian::{
    barrier_survival_bb, brownian_bridge_hit_prob, brownian_bridge_step, covariation,
    first_passage_time, ito_integral, quadratic_variation, running_maximum, running_minimum,
    CrossingDirection,
};
pub use greeks::{gbm_call_delta, pathwise_delta_call, GreekReport};
pub use hedging::{delta_hedge_call, HedgePath};
pub use jumps::{
    compensated_poisson, compound_poisson_increment, esscher_normal_jump, kou_call_mixture,
    merton_call, merton_call_mc, merton_call_pide, merton_compensator, CompensatedPoisson,
};
pub use market::{
    asset_forward, exponential_martingale, market_price_of_risk, t_forward_price, BondNumeraire,
    DiscountCurve, FlatCurve, MoneyMarket, Numeraire, PricingMeasure,
};
pub use monte_carlo::{
    price_ensemble, price_gbm_importance, price_sde, MonteCarloDiagnostics, MonteCarloEstimate,
    OnlineMoments, VarianceReduction,
};
pub use payoff::{
    AsianOption, BarrierActivation, BarrierDirection, BarrierOption, BasketCall, Digital,
    EuropeanCall, EuropeanPut, LookbackOption, PathPayoff, TerminalPayoff,
};
pub use pde::{
    american_put_psor, black_scholes_fd, cn_call_error, pde_value_at, solve_tridiagonal,
    BoundaryCondition, PdeGrid, PdeSolution, TimeScheme,
};
pub use rates::{
    black_caplet, black_swaption, cir_bond, forward_rate, hull_white_theta, vasicek_bond,
    vasicek_bond_option, BondQuote,
};
pub use realoption::{
    abandonment_option, entry_exit, finite_horizon_investment, mcdonald_siegel, EntryExit,
    PerpetualCall, PerpetualPut,
};
pub use stopping::{
    andersen_broadie_put, crr_storage, crr_swing, european_put_dual_upper, lsm_american_put, Basis,
    ConditionalExpectation, HermiteBasis, LinearCe, LongstaffSchwartzConfig, OptimalStoppingResult,
    PolynomialBasis, RegressionPolicy,
};
pub use tree::{
    crr_price, crr_vs_bs_call, replicate, state_prices, trinomial_price, CrrPrice, OnePeriodMarket,
    Portfolio,
};
