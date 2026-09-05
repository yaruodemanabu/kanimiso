#![allow(clippy::float_cmp)]

use riko::{ExpWeights, Ucb};

#[test]
fn ucb_initializes_every_arm_and_matches_closed_form_choice() {
    let mut policy = Ucb::new(3, 2.0_f64.sqrt()).unwrap();
    for reward in [0.0, 1.0, 0.5] {
        let choice = policy.select();
        policy.update(choice, reward).unwrap();
    }
    assert_eq!(policy.pulls(), [1, 1, 1]);
    // Equal bonuses leave empirical mean as the ordering criterion.
    assert_eq!(policy.select().arm(), 1);
    assert_eq!(policy.mean_reward(1), Some(1.0));
}

#[test]
fn exponential_weights_matches_a_hand_computed_update() {
    let mut policy = ExpWeights::new(2, 0.2).unwrap();
    assert_eq!(policy.probabilities(), [0.5, 0.5]);
    let choice = policy.select(0.25).unwrap();
    policy.update(choice, 1.0).unwrap();
    let probability = (1.0 - 0.2) * 0.2_f64.exp() / (0.2_f64.exp() + 1.0) + 0.1;
    // Closed-form one-step replay; measured absolute error is 0, tolerance is 0.
    assert_eq!(policy.probabilities()[0], probability);
}

#[test]
fn distributions_are_normalized_and_sampling_respects_boundaries() {
    let policy = ExpWeights::new(3, 0.1).unwrap();
    let p = policy.probabilities();
    assert_eq!(p.iter().sum::<f64>(), 1.0);
    assert_eq!(policy.select(0.0).unwrap().arm(), 0);
    assert_eq!(policy.select(p[0]).unwrap().arm(), 1);
    assert_eq!(policy.select(1.0 - f64::EPSILON).unwrap().arm(), 2);
}

#[test]
fn stale_and_invalid_feedback_do_not_advance_policy() {
    let mut policy = ExpWeights::new(2, 0.1).unwrap();
    let choice = policy.select(0.0).unwrap();
    assert!(policy.update(choice, f64::NAN).is_err());
    assert_eq!(policy.rounds(), 0);
    policy.update(choice, 1.0).unwrap();
    assert!(policy.update(choice, 1.0).is_err());
    assert_eq!(policy.rounds(), 1);
}
