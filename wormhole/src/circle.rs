//! Optimal transport on the unit circle.

use crate::error::{Error, Result};
use crate::validate;

fn prepare(
    locations: &[f64],
    weights: Option<&[f64]>,
    name: &'static str,
) -> Result<(Vec<f64>, Vec<f64>)> {
    if locations.is_empty() {
        return Err(Error::EmptyInput { name });
    }
    for (index, &location) in locations.iter().enumerate() {
        if !location.is_finite() {
            return Err(Error::InvalidCost {
                row: 0,
                column: index,
                value: location,
            });
        }
    }
    let owned;
    let weights = match weights {
        Some(weights) => weights,
        None => {
            owned = validate::uniform(locations.len())?;
            &owned
        }
    };
    if weights.len() != locations.len() {
        return Err(Error::ShapeMismatch {
            context: "circle locations and weights",
            left: (locations.len(), 1),
            right: (weights.len(), 1),
        });
    }
    let mass = validate::distribution(weights, "circle weights")?;
    let mut order: Vec<_> = (0..locations.len()).collect();
    order.sort_by(|&left, &right| {
        locations[left]
            .rem_euclid(1.0)
            .total_cmp(&locations[right].rem_euclid(1.0))
    });
    Ok((
        order
            .iter()
            .map(|&index| locations[index].rem_euclid(1.0))
            .collect(),
        order.iter().map(|&index| weights[index] / mass).collect(),
    ))
}

fn wasserstein_one_circle(
    source_locations: &[f64],
    target_locations: &[f64],
    source_weights: &[f64],
    target_weights: &[f64],
) -> f64 {
    let mut events = Vec::with_capacity(source_locations.len() + target_locations.len());
    for (&location, &weight) in source_locations.iter().zip(source_weights) {
        events.push((location, weight));
    }
    for (&location, &weight) in target_locations.iter().zip(target_weights) {
        events.push((location, -weight));
    }
    events.sort_by(|left, right| left.0.total_cmp(&right.0));
    let mut levels = Vec::<(f64, f64)>::new();
    let mut cumulative = 0.0;
    let mut position = 0.0;
    let mut index = 0;
    while index < events.len() {
        let event_position = events[index].0;
        if event_position > position {
            levels.push((cumulative, event_position - position));
        }
        while index < events.len() && events[index].0.to_bits() == event_position.to_bits() {
            cumulative += events[index].1;
            index += 1;
        }
        position = event_position;
    }
    if position < 1.0 {
        levels.push((cumulative, 1.0 - position));
    }
    let mut sorted = levels.clone();
    sorted.sort_by(|left, right| left.0.total_cmp(&right.0));
    let mut accumulated = 0.0;
    let mut median = 0.0;
    for &(level, interval) in &sorted {
        accumulated += interval;
        median = level;
        if accumulated >= 0.5 {
            break;
        }
    }
    levels
        .iter()
        .map(|&(level, interval)| interval * (level - median).abs())
        .sum()
}

fn cumulative(weights: &[f64]) -> Vec<f64> {
    let mut sum = 0.0;
    weights
        .iter()
        .map(|&weight| {
            sum += weight;
            sum
        })
        .collect()
}

fn lifted_cost(
    theta: f64,
    source_locations: &[f64],
    target_locations: &[f64],
    source_weights: &[f64],
    target_weights: &[f64],
    p: f64,
) -> f64 {
    let source_cumulative = cumulative(source_weights);
    let target_cumulative = cumulative(target_weights);
    let mut value = 0.0;
    let mut source_start: f64 = 0.0;
    for (source_index, &source_end) in source_cumulative.iter().enumerate() {
        let mut target_start: f64 = 0.0;
        for (target_index, &target_end) in target_cumulative.iter().enumerate() {
            for lift in -2_i32..=2 {
                let shifted_start = target_start - theta + lift as f64;
                let shifted_end = target_end - theta + lift as f64;
                let overlap_start = source_start.max(shifted_start).max(0.0);
                let overlap_end = source_end.min(shifted_end).min(1.0);
                if overlap_end > overlap_start {
                    let target_location = target_locations[target_index] + lift as f64;
                    value += (overlap_end - overlap_start)
                        * (source_locations[source_index] - target_location)
                            .abs()
                            .powf(p);
                }
            }
            target_start = target_end;
        }
        source_start = source_end;
    }
    value
}

/// Circular Wasserstein objective `W_p^p` for atoms represented in `[0, 1)`.
///
/// Input locations are reduced modulo one.  The `p=1` path uses the exact
/// level-median formula; larger exponents minimize the lifted quantile cost.
pub fn wasserstein_circle(
    source_locations: &[f64],
    target_locations: &[f64],
    source_weights: Option<&[f64]>,
    target_weights: Option<&[f64]>,
    p: f64,
) -> Result<f64> {
    if !p.is_finite() || p < 1.0 {
        return Err(Error::InvalidParameter {
            name: "p",
            requirement: "finite and at least one",
        });
    }
    let (source_locations, source_weights) =
        prepare(source_locations, source_weights, "source circle locations")?;
    let (target_locations, target_weights) =
        prepare(target_locations, target_weights, "target circle locations")?;
    if p == 1.0 {
        return Ok(wasserstein_one_circle(
            &source_locations,
            &target_locations,
            &source_weights,
            &target_weights,
        ));
    }
    let objective = |theta| {
        lifted_cost(
            theta,
            &source_locations,
            &target_locations,
            &source_weights,
            &target_weights,
            p,
        )
    };
    // The lifted cost is convex over a period for convex Monge costs.  A
    // golden-section minimization avoids differentiability assumptions at atoms.
    let ratio = (5.0_f64.sqrt() - 1.0) / 2.0;
    let (mut left, mut right) = (-1.0, 1.0);
    let mut first = right - ratio * (right - left);
    let mut second = left + ratio * (right - left);
    let mut first_value = objective(first);
    let mut second_value = objective(second);
    for _ in 0..100 {
        if first_value <= second_value {
            right = second;
            second = first;
            second_value = first_value;
            first = right - ratio * (right - left);
            first_value = objective(first);
        } else {
            left = first;
            first = second;
            first_value = second_value;
            second = left + ratio * (right - left);
            second_value = objective(second);
        }
    }
    Ok(first_value
        .min(second_value)
        .min(objective(-1.0))
        .min(objective(0.0))
        .min(objective(1.0)))
}

/// Closed-form squared `W₂` from a discrete circle measure to uniform measure.
pub fn semidiscrete_wasserstein2_uniform_circle(
    locations: &[f64],
    weights: Option<&[f64]>,
) -> Result<f64> {
    let (locations, weights) = prepare(locations, weights, "circle locations")?;
    let mut cumulative_before = 0.0;
    let mut second_moment = 0.0;
    let mut mean = 0.0;
    let mut correction = 0.0;
    for (&location, &weight) in locations.iter().zip(&weights) {
        second_moment += weight * location * location;
        mean += weight * location;
        correction += location * weight * (1.0 - weight - 2.0 * cumulative_before);
        cumulative_before += weight;
    }
    Ok((second_moment - mean * mean + correction + 1.0 / 12.0).max(0.0))
}

fn quantile(probability: f64, locations: &[f64], weights: &[f64]) -> f64 {
    if probability <= 0.0 {
        return locations[0];
    }
    let mut cumulative = 0.0;
    for (index, &weight) in weights.iter().enumerate() {
        cumulative += weight;
        if probability <= cumulative + 1e-15 {
            // POT's circular embedding calls quantile_function with a leading
            // zero in the cumulative array; preserve that public convention.
            return locations[(index + 1).min(locations.len() - 1)];
        }
    }
    *locations.last().unwrap_or(&0.0)
}

/// Evaluate the linear circular-OT embedding against uniform measure.
pub fn linear_circular_embedding(
    evaluation_points: &[f64],
    locations: &[f64],
    weights: Option<&[f64]>,
) -> Result<Vec<f64>> {
    let (locations, weights) = prepare(locations, weights, "circle locations")?;
    let mean = locations
        .iter()
        .zip(&weights)
        .map(|(&location, &weight)| location * weight)
        .sum::<f64>();
    let mut output = Vec::with_capacity(evaluation_points.len());
    for (index, &point) in evaluation_points.iter().enumerate() {
        if !point.is_finite() {
            return Err(Error::InvalidCost {
                row: 0,
                column: index,
                value: point,
            });
        }
        let point = point.rem_euclid(1.0);
        let probability = (point - mean + 0.5).rem_euclid(1.0);
        output.push((quantile(probability, &locations, &weights) - point).rem_euclid(1.0));
    }
    Ok(output)
}

/// Linear circular OT squared distance using a 100-point uniform quadrature.
pub fn linear_circular_ot(
    source_locations: &[f64],
    target_locations: Option<&[f64]>,
    source_weights: Option<&[f64]>,
    target_weights: Option<&[f64]>,
) -> Result<f64> {
    let grid = (0..100)
        .map(|index| index as f64 / 100.0)
        .collect::<Vec<_>>();
    let source = linear_circular_embedding(&grid, source_locations, source_weights)?;
    match target_locations {
        Some(target_locations) => {
            let target = linear_circular_embedding(&grid, target_locations, target_weights)?;
            Ok(source
                .iter()
                .zip(target)
                .map(|(&left, right)| {
                    let difference = (left - right).abs();
                    difference.min(1.0 - difference).powi(2)
                })
                .sum::<f64>()
                / grid.len() as f64)
        }
        None => Ok(source
            .iter()
            .map(|&value| value.abs().min(1.0 - value.abs()).powi(2))
            .sum::<f64>()
            / grid.len() as f64),
    }
}

/// Convert a point on the Euclidean unit circle to its `[0, 1)` coordinate.
pub fn coordinate(x: f64, y: f64) -> Result<f64> {
    if !x.is_finite() || !y.is_finite() {
        return Err(Error::InvalidParameter {
            name: "circle coordinate",
            requirement: "finite",
        });
    }
    Ok((std::f64::consts::PI + (-y).atan2(-x)) / std::f64::consts::TAU)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraparound_diracs_are_close() {
        let first = wasserstein_circle(&[0.9], &[0.1], None, None, 1.0).unwrap();
        let second = wasserstein_circle(&[0.1], &[0.9], None, None, 2.0).unwrap();
        assert!((first - 0.2).abs() < 1e-12);
        assert!((second - 0.04).abs() < 1e-9, "value={second}");
    }

    #[test]
    fn pot_documentation_example_matches() {
        let source = [0.2, 0.5, 0.8];
        let target = [0.4, 0.5, 0.7];
        assert!(
            (wasserstein_circle(&source, &target, None, None, 1.0).unwrap() - 0.1).abs() < 1e-12
        );
        assert!(
            (linear_circular_ot(&source, Some(&target), None, None).unwrap() - 0.0127).abs()
                < 1e-12
        );
    }

    #[test]
    fn semidiscrete_documentation_example_matches() {
        let value = semidiscrete_wasserstein2_uniform_circle(&[0.0, 0.2, 0.4], None).unwrap();
        assert!((value - 0.02111111111111111).abs() < 1e-12);
    }
}
