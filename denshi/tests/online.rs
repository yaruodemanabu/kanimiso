#![allow(clippy::float_cmp)]

use denshi::{Hedge, OnlineGradientDescent};

#[test]
fn hedge_matches_a_hand_computed_two_round_trace() {
    let mut learner = Hedge::new(2, 1.0).unwrap();
    assert_eq!(learner.probabilities(), [0.5, 0.5]);
    assert_eq!(learner.update(&[0.0, 1.0]).unwrap(), 0.5);
    let p = learner.probabilities();
    let expected = 1.0 / (1.0 + (-1.0_f64).exp());
    // Closed-form softmax trace; measured absolute error is 0, tolerance is 0.
    assert_eq!(p[0], expected);
    assert_eq!(p[0] + p[1], 1.0);
    let second = learner.update(&[0.0, 1.0]).unwrap();
    assert_eq!(second, p[1]);
    assert!(learner.regret() >= 0.0);
}

#[test]
fn hedge_probability_properties_hold_under_extreme_cumulative_loss() {
    let mut learner = Hedge::new(3, 100.0).unwrap();
    for _ in 0..100 {
        learner.update(&[0.0, 1.0, 0.5]).unwrap();
    }
    let probabilities = learner.probabilities();
    assert!(probabilities
        .iter()
        .all(|p| p.is_finite() && *p >= 0.0 && *p <= 1.0));
    assert_eq!(probabilities.iter().sum::<f64>(), 1.0);
    assert_eq!(learner.expert_losses(), [0.0, 100.0, 50.0]);
}

#[test]
fn projected_gradient_step_matches_closed_form_and_stays_feasible() {
    let mut learner = OnlineGradientDescent::new(2, 1.0, 1.0).unwrap();
    learner.update(&[-3.0, -4.0]).unwrap();
    // Projection is (0.6, 0.8). Measured error 1.11e-16; tolerance is 4.5e-16.
    assert!((learner.decision()[0] - 0.6).abs() <= 4.5e-16);
    assert_eq!(learner.decision()[1], 0.8);
    learner.update(&[-30.0, 40.0]).unwrap();
    let norm = learner.decision().iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(norm <= 1.0);
    assert_eq!(learner.cumulative_gradient(), [-33.0, 36.0]);
}

#[test]
fn invalid_feedback_is_rejected_without_mutation() {
    let mut hedge = Hedge::new(2, 0.5).unwrap();
    assert!(hedge.update(&[0.0, f64::NAN]).is_err());
    assert_eq!(hedge.rounds(), 0);
    let mut gradient = OnlineGradientDescent::new(2, 0.5, 1.0).unwrap();
    assert!(gradient.update(&[1.0]).is_err());
    assert_eq!(gradient.rounds(), 0);

    let original = gradient.decision().to_vec();
    let mut overflowing = OnlineGradientDescent::new(2, 2.0, 1.0).unwrap();
    assert!(overflowing.update(&[f64::MAX, 1.0]).is_err());
    assert_eq!(overflowing.decision(), original);
    assert_eq!(overflowing.rounds(), 0);
}
