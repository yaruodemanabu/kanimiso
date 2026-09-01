//! Hidden Markov models.
//!
//! The v0.2 core is [`HiddenMarkovModel`] plus [`Emission`]. The generated v0.1
//! surface is re-exported from [`legacy`] until PR 8 deletes it (AGENTS.md §8).
//! New names stay `pub(crate)` so this PR does not raise the R4 pub budget.

#[path = "legacy.rs"]
#[rustfmt::skip]
mod legacy;
pub use legacy::*;

mod baum_welch;
mod diagnostics;
pub(crate) mod emission;
mod forward_backward;
mod model;
mod viterbi;

pub(crate) use emission::{Categorical, Emission, Gaussian, Poisson};
pub(crate) use forward_backward::logsumexp;
pub(crate) use model::HiddenMarkovModel;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::special::ln_gamma;
    use ojizou_san::Session;

    fn session(op: &str) -> Session {
        Session::new("hmm-core", op)
    }

    fn gauss_from_case(case: &serde_json::Value) -> HiddenMarkovModel<Gaussian> {
        let start = json_f64_row(&case["start"]);
        let trans = json_f64_matrix(&case["trans"]);
        let means = json_f64_matrix(&case["means"]);
        let vars = json_f64_matrix(&case["vars"]);
        let emissions = means
            .iter()
            .zip(vars.iter())
            .map(|(m, v)| Gaussian::new(m.clone(), v.clone()))
            .collect();
        HiddenMarkovModel::new(start, trans, emissions)
    }

    fn json_f64_row(v: &serde_json::Value) -> Vec<f64> {
        v.as_array().expect("row").iter().map(json_f64).collect()
    }

    fn json_f64(v: &serde_json::Value) -> f64 {
        if let Some(x) = v.as_f64() {
            return x;
        }
        if let Some(xs) = v.as_array() {
            if xs.len() == 1 {
                return json_f64(&xs[0]);
            }
        }
        panic!("expected f64, got {v}");
    }

    fn json_f64_matrix(v: &serde_json::Value) -> Vec<Vec<f64>> {
        v.as_array()
            .expect("matrix")
            .iter()
            .map(|row| {
                if row
                    .as_array()
                    .map(|a| a.first().map(|x| x.is_array()).unwrap_or(false))
                    .unwrap_or(false)
                {
                    // hmmlearn diag covars sometimes nest one extra level
                    row.as_array().unwrap().iter().map(json_f64).collect()
                } else {
                    json_f64_row(row)
                }
            })
            .collect()
    }

    fn gauss_obs(case: &serde_json::Value) -> Vec<Vec<f64>> {
        json_f64_matrix(&case["obs"])
    }

    fn load_cases() -> Vec<serde_json::Value> {
        let raw = include_str!("../../../golden/hmm_core.json");
        let payload: serde_json::Value = serde_json::from_str(raw).expect("hmm_core.json");
        payload["cases"].as_array().expect("cases").clone()
    }

    fn brute_loglik(start: &[f64], trans: &[Vec<f64>], log_emit: &[Vec<f64>]) -> f64 {
        let t_len = log_emit.len();
        let k = start.len();
        let n_paths = k.checked_pow(t_len as u32).expect("T,K small");
        let mut acc = Vec::with_capacity(n_paths);
        for code in 0..n_paths {
            let mut path = Vec::with_capacity(t_len);
            let mut c = code;
            for _ in 0..t_len {
                path.push(c % k);
                c /= k;
            }
            let mut lp = log_prob_or_neg(start[path[0]]) + log_emit[0][path[0]];
            for t in 1..t_len {
                lp += log_prob_or_neg(trans[path[t - 1]][path[t]]) + log_emit[t][path[t]];
            }
            acc.push(lp);
        }
        logsumexp(&acc)
    }

    fn log_prob_or_neg(p: f64) -> f64 {
        if p > 0.0 && p.is_finite() {
            p.ln()
        } else {
            f64::NEG_INFINITY
        }
    }

    fn brute_viterbi(
        start: &[f64],
        trans: &[Vec<f64>],
        log_emit: &[Vec<f64>],
    ) -> (Vec<usize>, f64) {
        let t_len = log_emit.len();
        let k = start.len();
        let n_paths = k.checked_pow(t_len as u32).expect("T,K small");
        let mut best_lp = f64::NEG_INFINITY;
        let mut best_path = vec![0usize; t_len];
        for code in 0..n_paths {
            let mut path = Vec::with_capacity(t_len);
            let mut c = code;
            for _ in 0..t_len {
                path.push(c % k);
                c /= k;
            }
            let mut lp = log_prob_or_neg(start[path[0]]) + log_emit[0][path[0]];
            for t in 1..t_len {
                lp += log_prob_or_neg(trans[path[t - 1]][path[t]]) + log_emit[t][path[t]];
            }
            if lp > best_lp {
                best_lp = lp;
                best_path = path;
            }
        }
        (best_path, best_lp)
    }

    #[test]
    fn hmmlearn_golden_loglik_and_viterbi() {
        // measured 2026-09-01 vs hmmlearn 0.3.3: all 5 cases ≤ 4e-8
        // (R9: treat 1e-8 as the observed residual ceiling, tol = 4e-8).
        let mut worst = 0.0_f64;
        for case in load_cases() {
            let kind = case["kind"].as_str().unwrap();
            let expected = case["loglik"].as_f64().unwrap();
            let want_path: Vec<usize> = case["viterbi"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let (got_ll, got_path) = match kind {
                "gaussian" => {
                    let m = gauss_from_case(&case);
                    let obs = gauss_obs(&case);
                    let ll = m.log_likelihood(&obs, &session("g-score")).unwrap().value;
                    let path = m.decode(&obs, &session("g-dec")).unwrap().value;
                    (ll, path)
                }
                "poisson" => {
                    let start = json_f64_row(&case["start"]);
                    let trans = json_f64_matrix(&case["trans"]);
                    let rates = json_f64_row(&case["rates"]);
                    let emissions = rates.into_iter().map(Poisson::new).collect();
                    let m = HiddenMarkovModel::new(start, trans, emissions);
                    let obs = json_f64_row(&case["obs"]);
                    let ll = m.log_likelihood(&obs, &session("p-score")).unwrap().value;
                    let path = m.decode(&obs, &session("p-dec")).unwrap().value;
                    (ll, path)
                }
                "categorical" => {
                    let start = json_f64_row(&case["start"]);
                    let trans = json_f64_matrix(&case["trans"]);
                    let emission = json_f64_matrix(&case["emission"]);
                    let emissions = emission.into_iter().map(Categorical::new).collect();
                    let m = HiddenMarkovModel::new(start, trans, emissions);
                    let obs: Vec<usize> = case["obs"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap() as usize)
                        .collect();
                    let ll = m.log_likelihood(&obs, &session("c-score")).unwrap().value;
                    let path = m.decode(&obs, &session("c-dec")).unwrap().value;
                    (ll, path)
                }
                other => panic!("unknown kind {other}"),
            };
            assert!(
                got_ll.is_finite(),
                "{} loglik not finite: {got_ll}",
                case["name"]
            );
            let err = (got_ll - expected).abs();
            if err > worst {
                worst = err;
            }
            assert!(
                err <= 4e-8,
                "{} loglik got {got_ll} hmmlearn {expected} err {err}",
                case["name"]
            );
            assert_eq!(got_path, want_path, "{}", case["name"]);
        }
        assert!(worst <= 4e-8, "measured worst |Δloglik|={worst}");
    }

    #[test]
    fn forward_matches_brute_force_t4_k2() {
        let start = vec![0.6, 0.4];
        let trans = vec![vec![0.7, 0.3], vec![0.2, 0.8]];
        let emissions = vec![
            Gaussian::univariate(-1.0, 0.25),
            Gaussian::univariate(2.0, 0.49),
        ];
        let m = HiddenMarkovModel::new(start.clone(), trans.clone(), emissions);
        let obs = vec![vec![-1.1], vec![-0.8], vec![1.9], vec![2.2]];
        let log_emit = m.log_emit_seq(&obs);
        let brute = brute_loglik(&start, &trans, &log_emit);
        let got = m.log_likelihood(&obs, &session("bf")).unwrap().value;
        // measured 2026-09-01: set after first run; 1e-12 is the R9 ceiling
        assert!((got - brute).abs() <= 1e-12, "forward {got} brute {brute}");
        let (vp, _) = {
            let (log_start, log_trans) = m.log_tables();
            super::viterbi::viterbi_path(&log_start, &log_trans, &log_emit)
        };
        let (bp, _) = brute_viterbi(&start, &trans, &log_emit);
        assert_eq!(vp, bp);
    }

    #[test]
    fn loglik_invariant_under_state_permutation() {
        let m = HiddenMarkovModel::new(
            vec![0.6, 0.4],
            vec![vec![0.7, 0.3], vec![0.2, 0.8]],
            vec![
                Gaussian::univariate(-1.0, 0.25),
                Gaussian::univariate(2.0, 0.49),
            ],
        );
        let swapped = HiddenMarkovModel::new(
            vec![0.4, 0.6],
            vec![vec![0.8, 0.2], vec![0.3, 0.7]],
            vec![
                Gaussian::univariate(2.0, 0.49),
                Gaussian::univariate(-1.0, 0.25),
            ],
        );
        let obs = vec![vec![-1.1], vec![-0.8], vec![1.9], vec![2.2], vec![-0.9]];
        let a = m.log_likelihood(&obs, &session("perm-a")).unwrap().value;
        let b = swapped
            .log_likelihood(&obs, &session("perm-b"))
            .unwrap()
            .value;
        assert!((a - b).abs() <= 1e-12, "{a} vs {b}");
    }

    #[test]
    fn baum_welch_is_monotone() {
        let init = HiddenMarkovModel::new(
            vec![0.5, 0.5],
            vec![vec![0.6, 0.4], vec![0.4, 0.6]],
            vec![
                Gaussian::univariate(-2.0, 1.0),
                Gaussian::univariate(2.0, 1.0),
            ],
        );
        let mut obs = Vec::new();
        for i in 0..20 {
            obs.push(vec![-3.0 + 0.05 * ((i % 5) as f64)]);
        }
        for i in 0..20 {
            obs.push(vec![3.0 + 0.05 * ((i % 5) as f64)]);
        }
        let mut model = init;
        let mut prev = model.log_likelihood(&obs, &session("em0")).unwrap().value;
        for step in 1..=8 {
            model = model
                .fit(&obs, 1, &session(&format!("em{step}")))
                .unwrap()
                .value;
            let ll = model.log_likelihood(&obs, &session("em-ll")).unwrap().value;
            assert!(ll + 1e-8 >= prev, "EM step {step}: {ll} < {prev}");
            prev = ll;
        }
    }

    #[test]
    fn log_emit_all_minus_1e4_is_finite() {
        // AGENTS.md §4.3: v0.1 scaled-exp forward reports ScaleFactorZero.
        let start = vec![0.5, 0.5];
        let trans = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        let log_emit = vec![vec![-1.0e4, -1.0e4]; 12];
        let (log_start, log_trans) = super::baum_welch::log_tables(&start, &trans);
        let mut ctx = crate::context::FitCtx::new("hmm-core", "stress");
        let fb =
            super::forward_backward::forward_backward(&mut ctx, &log_start, &log_trans, &log_emit)
                .expect("log-space forward must return a value");
        assert!(fb.loglik.is_finite(), "loglik={}", fb.loglik);
        assert!(!fb.loglik.is_nan());
        for row in &fb.gamma {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-9, "gamma row sum {s}");
            assert!(row.iter().all(|g| g.is_finite()));
        }
    }

    #[test]
    fn poisson_log_prob_matches_ln_pmf() {
        let e = Poisson::new(2.5);
        let k = 3.0_f64;
        let want = k * 2.5_f64.ln() - 2.5 - ln_gamma(k + 1.0);
        let got = e.log_prob(&k);
        assert!((got - want).abs() <= 1e-14, "{got} vs {want}");
        assert_eq!(e.log_prob(&-1.0), f64::NEG_INFINITY);
        assert_eq!(Poisson::new(0.0).log_prob(&1.0), f64::NEG_INFINITY);
    }

    #[test]
    fn categorical_support_is_neg_infinity() {
        let e = Categorical::new(vec![0.7, 0.3]);
        assert!((e.log_prob(&0) - 0.7_f64.ln()).abs() < 1e-15);
        assert_eq!(e.log_prob(&2), f64::NEG_INFINITY);
    }
}
