//! Exact balanced discrete optimal transport and one-dimensional solvers.

use crate::error::{Error, Result};
use crate::result::{SolverStatus, TransportPlan};
use crate::validate;
use faer::Mat;

/// Numerical options for the exact min-cost-flow solver.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmdOptions {
    /// Residual capacity at or below this scale is treated as zero.
    pub capacity_tolerance: f64,
    /// Maximum number of augmenting paths; `None` selects a size-based bound.
    pub max_augmentations: Option<usize>,
}

impl Default for EmdOptions {
    fn default() -> Self {
        Self {
            capacity_tolerance: 1e-12,
            max_augmentations: None,
        }
    }
}

#[derive(Clone, Debug)]
struct Edge {
    to: usize,
    reverse: usize,
    capacity: f64,
    cost: f64,
}

fn add_edge(graph: &mut [Vec<Edge>], from: usize, to: usize, capacity: f64, cost: f64) {
    let forward_index = graph[from].len();
    let reverse_index = graph[to].len();
    graph[from].push(Edge {
        to,
        reverse: reverse_index,
        capacity,
        cost,
    });
    graph[to].push(Edge {
        to: from,
        reverse: forward_index,
        capacity: 0.0,
        cost: -cost,
    });
}

/// Solve the balanced Earth Mover's Distance problem.
///
/// This Pure-Rust implementation applies successive shortest augmenting paths
/// to the complete bipartite transport network.  It supports arbitrary finite
/// costs and fractional masses.
pub fn emd(source: &[f64], target: &[f64], cost: &Mat<f64>) -> Result<TransportPlan> {
    emd_with_options(source, target, cost, EmdOptions::default())
}

/// Solve balanced EMD with explicit numerical options.
pub fn emd_with_options(
    source: &[f64],
    target: &[f64],
    cost: &Mat<f64>,
    options: EmdOptions,
) -> Result<TransportPlan> {
    let total_mass = validate::balanced_distributions(source, target)?;
    validate::cost_matrix(cost, source.len(), target.len())?;
    validate::finite_positive(
        options.capacity_tolerance,
        "capacity_tolerance",
        "finite and strictly positive",
    )?;

    let source_node = 0;
    let first_source_atom = 1;
    let first_target_atom = first_source_atom + source.len();
    let sink_node = first_target_atom + target.len();
    let node_count = sink_node + 1;
    let mut graph = vec![Vec::<Edge>::new(); node_count];
    for (i, &mass) in source.iter().enumerate() {
        add_edge(&mut graph, source_node, first_source_atom + i, mass, 0.0);
    }
    let mut plan_edges = vec![vec![0_usize; target.len()]; source.len()];
    for i in 0..source.len() {
        for j in 0..target.len() {
            let node = first_source_atom + i;
            plan_edges[i][j] = graph[node].len();
            add_edge(
                &mut graph,
                node,
                first_target_atom + j,
                total_mass,
                cost[(i, j)],
            );
        }
    }
    for (j, &mass) in target.iter().enumerate() {
        add_edge(&mut graph, first_target_atom + j, sink_node, mass, 0.0);
    }

    let default_limit = source
        .len()
        .saturating_mul(target.len())
        .saturating_mul(4)
        .saturating_add(source.len())
        .saturating_add(target.len())
        .max(1);
    let limit = options.max_augmentations.unwrap_or(default_limit);
    let mut flow = 0.0;
    let mut iterations = 0;
    while total_mass - flow > options.capacity_tolerance * total_mass.max(1.0) {
        if iterations >= limit {
            return Err(Error::DidNotConverge {
                algorithm: "exact EMD min-cost flow",
                iterations,
                residual: total_mass - flow,
            });
        }
        let mut distance = vec![f64::INFINITY; node_count];
        let mut predecessor = vec![None::<(usize, usize)>; node_count];
        distance[source_node] = 0.0;
        for _ in 0..node_count.saturating_sub(1) {
            let mut changed = false;
            for node in 0..node_count {
                if !distance[node].is_finite() {
                    continue;
                }
                for (edge_index, edge) in graph[node].iter().enumerate() {
                    if edge.capacity <= options.capacity_tolerance {
                        continue;
                    }
                    let candidate = distance[node] + edge.cost;
                    if candidate + 1e-14 < distance[edge.to] {
                        distance[edge.to] = candidate;
                        predecessor[edge.to] = Some((node, edge_index));
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        if predecessor[sink_node].is_none() {
            return Err(Error::Infeasible {
                context: "no residual source-to-target path",
            });
        }
        let mut augmentation = total_mass - flow;
        let mut node = sink_node;
        while node != source_node {
            let (previous, edge_index) = predecessor[node].ok_or(Error::Infeasible {
                context: "shortest-path predecessor chain is incomplete",
            })?;
            augmentation = augmentation.min(graph[previous][edge_index].capacity);
            node = previous;
        }
        if !augmentation.is_finite() || augmentation <= options.capacity_tolerance {
            return Err(Error::Infeasible {
                context: "augmenting path has no usable residual capacity",
            });
        }
        node = sink_node;
        while node != source_node {
            let (previous, edge_index) = predecessor[node].ok_or(Error::Infeasible {
                context: "augmenting path was lost",
            })?;
            let reverse_index = graph[previous][edge_index].reverse;
            graph[previous][edge_index].capacity -= augmentation;
            graph[node][reverse_index].capacity += augmentation;
            node = previous;
        }
        flow += augmentation;
        iterations += 1;
    }

    let mut plan = Mat::<f64>::zeros(source.len(), target.len());
    let mut value = 0.0;
    for i in 0..source.len() {
        let node = first_source_atom + i;
        for j in 0..target.len() {
            let edge = &graph[node][plan_edges[i][j]];
            let transported = graph[edge.to][edge.reverse].capacity;
            let transported = if transported.abs() <= options.capacity_tolerance {
                0.0
            } else {
                transported
            };
            plan[(i, j)] = transported;
            value += transported * cost[(i, j)];
        }
    }
    let residual = marginal_residual(&plan, source, target);
    Ok(TransportPlan {
        plan,
        value,
        potentials: None,
        iterations,
        residual,
        status: SolverStatus::Converged,
    })
}

/// Return only the balanced EMD objective value.
pub fn emd2(source: &[f64], target: &[f64], cost: &Mat<f64>) -> Result<f64> {
    Ok(emd(source, target, cost)?.value)
}

fn marginal_residual(plan: &Mat<f64>, source: &[f64], target: &[f64]) -> f64 {
    let mut residual = 0.0_f64;
    for i in 0..plan.nrows() {
        let row_sum = (0..plan.ncols()).map(|j| plan[(i, j)]).sum::<f64>();
        residual = residual.max((row_sum - source[i]).abs());
    }
    for j in 0..plan.ncols() {
        let column_sum = (0..plan.nrows()).map(|i| plan[(i, j)]).sum::<f64>();
        residual = residual.max((column_sum - target[j]).abs());
    }
    residual
}

fn validate_locations(locations: &[f64], name: &'static str) -> Result<()> {
    if locations.is_empty() {
        return Err(Error::EmptyInput { name });
    }
    for (index, value) in locations.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(Error::InvalidCost {
                row: 0,
                column: index,
                value,
            });
        }
    }
    Ok(())
}

/// Exact one-dimensional transport with cost `|x-y|^p`.
///
/// The returned plan uses the original, unsorted atom order.
pub fn emd_1d(
    source_locations: &[f64],
    target_locations: &[f64],
    source_weights: &[f64],
    target_weights: &[f64],
    p: f64,
) -> Result<TransportPlan> {
    validate_locations(source_locations, "source locations")?;
    validate_locations(target_locations, "target locations")?;
    if source_locations.len() != source_weights.len()
        || target_locations.len() != target_weights.len()
    {
        return Err(Error::ShapeMismatch {
            context: "one-dimensional locations and weights",
            left: (source_locations.len(), source_weights.len()),
            right: (target_locations.len(), target_weights.len()),
        });
    }
    validate::balanced_distributions(source_weights, target_weights)?;
    validate::finite_positive(p, "p", "finite and strictly positive")?;
    let mut source_order: Vec<_> = (0..source_locations.len()).collect();
    let mut target_order: Vec<_> = (0..target_locations.len()).collect();
    source_order.sort_by(|&i, &j| source_locations[i].total_cmp(&source_locations[j]));
    target_order.sort_by(|&i, &j| target_locations[i].total_cmp(&target_locations[j]));
    let mut source_remaining = source_weights.to_vec();
    let mut target_remaining = target_weights.to_vec();
    let mut plan = Mat::<f64>::zeros(source_locations.len(), target_locations.len());
    let (mut source_index, mut target_index) = (0_usize, 0_usize);
    let mut value = 0.0;
    let mut iterations = 0;
    while source_index < source_order.len() && target_index < target_order.len() {
        let i = source_order[source_index];
        let j = target_order[target_index];
        let transported = source_remaining[i].min(target_remaining[j]);
        if transported > 0.0 {
            plan[(i, j)] += transported;
            value += transported * (source_locations[i] - target_locations[j]).abs().powf(p);
            source_remaining[i] -= transported;
            target_remaining[j] -= transported;
            iterations += 1;
        }
        if source_remaining[i] <= 1e-14 {
            source_index += 1;
        }
        if target_remaining[j] <= 1e-14 {
            target_index += 1;
        }
    }
    let residual = marginal_residual(&plan, source_weights, target_weights);
    Ok(TransportPlan {
        plan,
        value,
        potentials: None,
        iterations,
        residual,
        status: SolverStatus::Converged,
    })
}

/// One-dimensional Wasserstein objective `integral |F⁻¹-G⁻¹|^p`.
pub fn wasserstein_1d(
    source_locations: &[f64],
    target_locations: &[f64],
    source_weights: Option<&[f64]>,
    target_weights: Option<&[f64]>,
    p: f64,
) -> Result<f64> {
    let owned_source;
    let owned_target;
    let source_weights = match source_weights {
        Some(weights) => weights,
        None => {
            owned_source = validate::uniform(source_locations.len())?;
            &owned_source
        }
    };
    let target_weights = match target_weights {
        Some(weights) => weights,
        None => {
            owned_target = validate::uniform(target_locations.len())?;
            &owned_target
        }
    };
    Ok(emd_1d(
        source_locations,
        target_locations,
        source_weights,
        target_weights,
        p,
    )?
    .value)
}

/// Evaluate the left-continuous empirical quantile function.
pub fn quantiles(probabilities: &[f64], locations: &[f64], weights: &[f64]) -> Result<Vec<f64>> {
    validate_locations(locations, "quantile locations")?;
    if locations.len() != weights.len() {
        return Err(Error::ShapeMismatch {
            context: "quantile locations and weights",
            left: (1, locations.len()),
            right: (1, weights.len()),
        });
    }
    let mass = validate::distribution(weights, "quantile weights")?;
    let mut order: Vec<_> = (0..locations.len()).collect();
    order.sort_by(|&i, &j| locations[i].total_cmp(&locations[j]));
    let mut cumulative = Vec::with_capacity(order.len());
    let mut sum = 0.0;
    for &index in &order {
        sum += weights[index] / mass;
        cumulative.push(sum);
    }
    let mut output = Vec::with_capacity(probabilities.len());
    for (index, &probability) in probabilities.iter().enumerate() {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(Error::InvalidWeight {
                index,
                value: probability,
            });
        }
        let position = cumulative
            .partition_point(|&value| value + 1e-15 < probability)
            .min(order.len() - 1);
        output.push(locations[order[position]]);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_solver_uses_zero_cost_diagonal() {
        let source = [0.5, 0.5];
        let target = [0.5, 0.5];
        let cost = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 0.0 } else { 1.0 });
        let result = emd(&source, &target, &cost).expect("feasible EMD");
        assert!(result.value.abs() < 1e-12);
        assert!((result.plan[(0, 0)] - 0.5).abs() < 1e-12);
        assert!((result.plan[(1, 1)] - 0.5).abs() < 1e-12);
        assert!(result.residual < 1e-12);
    }

    #[test]
    fn exact_solver_can_reroute_residual_flow() {
        let source = [0.5, 0.5];
        let target = [0.5, 0.5];
        let cost = Mat::<f64>::from_fn(2, 2, |i, j| [[1.0, 2.0], [2.0, 100.0]][i][j]);
        let result = emd(&source, &target, &cost).expect("feasible EMD");
        assert!((result.value - 2.0).abs() < 1e-12);
        assert!((result.plan[(0, 1)] - 0.5).abs() < 1e-12);
        assert!((result.plan[(1, 0)] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn one_dimensional_wasserstein_matches_translation() {
        let left = [0.0, 1.0];
        let right = [2.0, 3.0];
        let value = wasserstein_1d(&left, &right, None, None, 1.0).unwrap();
        assert!((value - 2.0).abs() < 1e-12);
    }

    #[test]
    fn empirical_quantiles_are_order_invariant() {
        let values = [3.0, 1.0, 2.0];
        let weights = [0.25, 0.5, 0.25];
        let result = quantiles(&[0.25, 0.75, 1.0], &values, &weights).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }
}
