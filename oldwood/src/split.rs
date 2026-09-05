use core::cmp::Ordering;

use crate::criterion::{classification_impurity, regression_impurity};
use crate::numeric::{CompensatedSum, WeightedMoments};
use crate::{
    ClassificationCriterion, Error, MatrixView, RegressionCriterion, Result, SplitContext,
    SplitStrategy, TreeOptions,
};

#[derive(Clone, Copy, Debug)]
pub(crate) struct BestSplit {
    pub(crate) feature: usize,
    pub(crate) threshold: f64,
    pub(crate) gain: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct ClassSummary {
    pub(crate) counts: Vec<f64>,
    pub(crate) weight: f64,
    pub(crate) impurity: f64,
    pub(crate) prediction: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RegressionSummary {
    pub(crate) moments: WeightedMoments,
    pub(crate) impurity: f64,
}

pub(crate) struct ClassSplitRequest<'a, M> {
    pub(crate) matrix: &'a M,
    pub(crate) targets: &'a [usize],
    pub(crate) weights: &'a [f64],
    pub(crate) classes: &'a [usize],
    pub(crate) indices: &'a [usize],
    pub(crate) parent: &'a ClassSummary,
    pub(crate) root_weight: f64,
    pub(crate) criterion: ClassificationCriterion,
    pub(crate) options: &'a TreeOptions,
    pub(crate) context: SplitContext,
}

pub(crate) struct RegressionSplitRequest<'a, M> {
    pub(crate) matrix: &'a M,
    pub(crate) targets: &'a [f64],
    pub(crate) weights: &'a [f64],
    pub(crate) indices: &'a [usize],
    pub(crate) parent: RegressionSummary,
    pub(crate) root_weight: f64,
    pub(crate) criterion: RegressionCriterion,
    pub(crate) options: &'a TreeOptions,
    pub(crate) context: SplitContext,
}

pub(crate) fn class_summary(
    indices: &[usize],
    targets: &[usize],
    weights: &[f64],
    classes: &[usize],
    criterion: ClassificationCriterion,
) -> Result<ClassSummary> {
    let mut rows: Vec<(usize, f64)> = indices
        .iter()
        .map(|&row| (targets[row], weights[row]))
        .collect();
    rows.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.1.total_cmp(&right.1))
    });

    let mut count_sums = vec![CompensatedSum::default(); classes.len()];
    for (class, weight) in rows {
        let class_index = classes
            .binary_search(&class)
            .expect("training classes contain every positive-weight target");
        count_sums[class_index].add(weight, "classification weight accumulation")?;
    }
    let counts = totals(&count_sums, "classification weight accumulation")?;
    let weight = sum(&counts, "classification node weight")?;
    let impurity = classification_impurity(criterion, &counts)?;
    let prediction_index = count_sums
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.cmp(*right.1).then_with(|| right.0.cmp(&left.0)))
        .map_or(0, |(index, _)| index);
    Ok(ClassSummary {
        counts,
        weight,
        impurity,
        prediction: classes[prediction_index],
    })
}

pub(crate) fn regression_summary(
    indices: &[usize],
    targets: &[f64],
    weights: &[f64],
    criterion: RegressionCriterion,
) -> Result<RegressionSummary> {
    let mut rows: Vec<(f64, f64)> = indices
        .iter()
        .map(|&row| (targets[row], weights[row]))
        .collect();
    rows.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.total_cmp(&right.1))
    });
    let mut moments = WeightedMoments::default();
    for (target, weight) in rows {
        moments.add(target, weight)?;
    }
    let impurity = regression_impurity(criterion, moments);
    if !impurity.is_finite() {
        return Err(Error::NumericalOverflow {
            operation: "regression impurity",
        });
    }
    Ok(RegressionSummary { moments, impurity })
}

pub(crate) fn best_class_split<M: MatrixView, S: SplitStrategy + ?Sized>(
    request: &ClassSplitRequest<'_, M>,
    strategy: &mut S,
) -> Result<Option<BestSplit>> {
    let features = candidate_features(
        request.matrix.ncols(),
        request.options,
        request.context,
        strategy,
    )?;
    let mut best = None;
    for feature in features {
        consider_class_feature(request, feature, strategy, &mut best)?;
    }
    Ok(best)
}

fn consider_class_feature<M: MatrixView, S: SplitStrategy + ?Sized>(
    request: &ClassSplitRequest<'_, M>,
    feature: usize,
    strategy: &mut S,
    best: &mut Option<BestSplit>,
) -> Result<()> {
    let mut rows: Vec<ClassRow> = request
        .indices
        .iter()
        .map(|&row| ClassRow {
            value: request.matrix.get(row, feature),
            class: request
                .classes
                .binary_search(&request.targets[row])
                .expect("training classes contain every target"),
            weight: request.weights[row],
        })
        .collect();
    rows.sort_by(|left, right| {
        left.value
            .total_cmp(&right.value)
            .then_with(|| left.class.cmp(&right.class))
            .then_with(|| left.weight.total_cmp(&right.weight))
    });
    let groups = class_groups(&rows, request.classes.len())?;
    if groups.len() < 2 {
        return Ok(());
    }
    let values: Vec<f64> = groups.iter().map(|group| group.value).collect();
    let thresholds = candidate_thresholds(request.context, feature, &values, strategy)?;
    let (prefix_counts, prefix_samples) = class_prefix(&groups, request.classes.len())?;
    let (suffix_counts, suffix_samples) = class_suffix(&groups, request.classes.len())?;

    for threshold in thresholds {
        let boundary = values.partition_point(|value| *value <= threshold);
        if boundary == 0 || boundary == groups.len() {
            continue;
        }
        if prefix_samples[boundary] < request.options.min_samples_leaf
            || suffix_samples[boundary] < request.options.min_samples_leaf
        {
            continue;
        }
        let left_counts = totals(&prefix_counts[boundary], "left class weight")?;
        let right_counts = totals(&suffix_counts[boundary], "right class weight")?;
        let left_weight = sum(&left_counts, "left class weight")?;
        let right_weight = sum(&right_counts, "right class weight")?;
        if left_weight < request.options.min_weight_leaf
            || right_weight < request.options.min_weight_leaf
        {
            continue;
        }
        let left_impurity = classification_impurity(request.criterion, &left_counts)?;
        let right_impurity = classification_impurity(request.criterion, &right_counts)?;
        let gain = request.parent.impurity
            - (left_weight / request.parent.weight) * left_impurity
            - (right_weight / request.parent.weight) * right_impurity;
        consider_candidate(
            best,
            BestSplit {
                feature,
                threshold,
                gain,
            },
            request.options.min_impurity_decrease,
            request.parent.weight,
            request.root_weight,
        )?;
    }
    Ok(())
}

pub(crate) fn best_regression_split<M: MatrixView, S: SplitStrategy + ?Sized>(
    request: &RegressionSplitRequest<'_, M>,
    strategy: &mut S,
) -> Result<Option<BestSplit>> {
    let features = candidate_features(
        request.matrix.ncols(),
        request.options,
        request.context,
        strategy,
    )?;
    let mut best = None;
    for feature in features {
        consider_regression_feature(request, feature, strategy, &mut best)?;
    }
    Ok(best)
}

fn consider_regression_feature<M: MatrixView, S: SplitStrategy + ?Sized>(
    request: &RegressionSplitRequest<'_, M>,
    feature: usize,
    strategy: &mut S,
    best: &mut Option<BestSplit>,
) -> Result<()> {
    let mut rows: Vec<RegressionRow> = request
        .indices
        .iter()
        .map(|&row| RegressionRow {
            value: request.matrix.get(row, feature),
            target: request.targets[row],
            weight: request.weights[row],
        })
        .collect();
    rows.sort_by(|left, right| {
        left.value
            .total_cmp(&right.value)
            .then_with(|| left.target.total_cmp(&right.target))
            .then_with(|| left.weight.total_cmp(&right.weight))
    });
    let groups = regression_groups(&rows)?;
    if groups.len() < 2 {
        return Ok(());
    }
    let values: Vec<f64> = groups.iter().map(|group| group.value).collect();
    let thresholds = candidate_thresholds(request.context, feature, &values, strategy)?;
    let (prefix_moments, prefix_samples) = regression_prefix(&groups)?;
    let (suffix_moments, suffix_samples) = regression_suffix(&groups)?;

    for threshold in thresholds {
        let boundary = values.partition_point(|value| *value <= threshold);
        if boundary == 0 || boundary == groups.len() {
            continue;
        }
        if prefix_samples[boundary] < request.options.min_samples_leaf
            || suffix_samples[boundary] < request.options.min_samples_leaf
        {
            continue;
        }
        let left = prefix_moments[boundary];
        let right = suffix_moments[boundary];
        if left.weight < request.options.min_weight_leaf
            || right.weight < request.options.min_weight_leaf
        {
            continue;
        }
        let left_impurity = regression_impurity(request.criterion, left);
        let right_impurity = regression_impurity(request.criterion, right);
        let gain = request.parent.impurity
            - (left.weight / request.parent.moments.weight) * left_impurity
            - (right.weight / request.parent.moments.weight) * right_impurity;
        consider_candidate(
            best,
            BestSplit {
                feature,
                threshold,
                gain,
            },
            request.options.min_impurity_decrease,
            request.parent.moments.weight,
            request.root_weight,
        )?;
    }
    Ok(())
}

pub(crate) fn partition<M: MatrixView>(
    matrix: &M,
    indices: &[usize],
    feature: usize,
    threshold: f64,
) -> (Vec<usize>, Vec<usize>) {
    indices
        .iter()
        .copied()
        .partition(|&row| matrix.get(row, feature) <= threshold)
}

fn candidate_features<S: SplitStrategy + ?Sized>(
    columns: usize,
    options: &TreeOptions,
    context: SplitContext,
    strategy: &mut S,
) -> Result<Vec<usize>> {
    let mut features = Vec::new();
    strategy.features(context, columns, &mut features);
    if let Some(&feature) = features.iter().find(|&&feature| feature >= columns) {
        return Err(Error::InvalidStrategyFeature { feature, columns });
    }
    // Stable canonical ordering is part of SplitStrategy's contract.
    #[allow(clippy::stable_sort_primitive)]
    features.sort();
    features.dedup();
    features.truncate(options.feature_count(columns));
    Ok(features)
}

fn candidate_thresholds<S: SplitStrategy + ?Sized>(
    context: SplitContext,
    feature: usize,
    unique_values: &[f64],
    strategy: &mut S,
) -> Result<Vec<f64>> {
    let lower = unique_values[0];
    let upper = *unique_values.last().expect("two unique values");
    let mut thresholds = Vec::new();
    strategy.thresholds(context, feature, unique_values, &mut thresholds);
    if thresholds
        .iter()
        .any(|threshold| !(threshold.is_finite() && *threshold >= lower && *threshold < upper))
    {
        return Err(Error::InvalidStrategyThreshold { feature });
    }
    thresholds.sort_by(f64::total_cmp);
    thresholds.dedup_by(|left, right| same_float(*left, *right));
    Ok(thresholds)
}

fn consider_candidate(
    best: &mut Option<BestSplit>,
    candidate: BestSplit,
    minimum_gain: f64,
    node_weight: f64,
    root_weight: f64,
) -> Result<()> {
    if !candidate.gain.is_finite() {
        return Err(Error::NumericalOverflow {
            operation: "impurity decrease",
        });
    }
    let weighted_gain = (node_weight / root_weight) * candidate.gain;
    if !weighted_gain.is_finite() {
        return Err(Error::NumericalOverflow {
            operation: "root-weighted impurity decrease",
        });
    }
    if candidate.gain <= 0.0 || weighted_gain < minimum_gain {
        return Ok(());
    }
    let replace = best.is_none_or(|current| match candidate.gain.total_cmp(&current.gain) {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => {
            (candidate.feature, candidate.threshold) < (current.feature, current.threshold)
        }
    });
    if replace {
        *best = Some(candidate);
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct ClassRow {
    value: f64,
    class: usize,
    weight: f64,
}

struct ClassGroup {
    value: f64,
    samples: usize,
    counts: Vec<CompensatedSum>,
}

fn class_groups(rows: &[ClassRow], classes: usize) -> Result<Vec<ClassGroup>> {
    let mut groups: Vec<ClassGroup> = Vec::new();
    for row in rows {
        if groups
            .last()
            .is_none_or(|group| !same_float(group.value, row.value))
        {
            groups.push(ClassGroup {
                value: row.value,
                samples: 0,
                counts: vec![CompensatedSum::default(); classes],
            });
        }
        let group = groups.last_mut().expect("group inserted");
        group.samples = group
            .samples
            .checked_add(1)
            .ok_or(Error::NumericalOverflow {
                operation: "classification group sample count",
            })?;
        group.counts[row.class].add(row.weight, "classification group weight")?;
    }
    Ok(groups)
}

fn class_prefix(
    groups: &[ClassGroup],
    classes: usize,
) -> Result<(Vec<Vec<CompensatedSum>>, Vec<usize>)> {
    let mut counts = Vec::with_capacity(groups.len() + 1);
    let mut samples = Vec::with_capacity(groups.len() + 1);
    counts.push(vec![CompensatedSum::default(); classes]);
    samples.push(0usize);
    for group in groups {
        let mut next = counts.last().cloned().expect("prefix seed");
        add_counts(&mut next, &group.counts)?;
        counts.push(next);
        samples.push(checked_sample_add(
            *samples.last().expect("prefix sample seed"),
            group.samples,
            "classification sample count",
        )?);
    }
    Ok((counts, samples))
}

fn class_suffix(
    groups: &[ClassGroup],
    classes: usize,
) -> Result<(Vec<Vec<CompensatedSum>>, Vec<usize>)> {
    let mut counts = vec![vec![CompensatedSum::default(); classes]; groups.len() + 1];
    let mut samples = vec![0usize; groups.len() + 1];
    for index in (0..groups.len()).rev() {
        let (before, after) = counts.split_at_mut(index + 1);
        before[index].clone_from(&after[0]);
        add_counts(&mut counts[index], &groups[index].counts)?;
        samples[index] = checked_sample_add(
            samples[index + 1],
            groups[index].samples,
            "classification sample count",
        )?;
    }
    Ok((counts, samples))
}

#[derive(Clone, Copy)]
struct RegressionRow {
    value: f64,
    target: f64,
    weight: f64,
}

struct RegressionGroup {
    value: f64,
    samples: usize,
    moments: WeightedMoments,
}

fn regression_groups(rows: &[RegressionRow]) -> Result<Vec<RegressionGroup>> {
    let mut groups: Vec<RegressionGroup> = Vec::new();
    for row in rows {
        if groups
            .last()
            .is_none_or(|group| !same_float(group.value, row.value))
        {
            groups.push(RegressionGroup {
                value: row.value,
                samples: 0,
                moments: WeightedMoments::default(),
            });
        }
        let group = groups.last_mut().expect("group inserted");
        group.samples = group
            .samples
            .checked_add(1)
            .ok_or(Error::NumericalOverflow {
                operation: "regression group sample count",
            })?;
        group.moments.add(row.target, row.weight)?;
    }
    Ok(groups)
}

fn regression_prefix(groups: &[RegressionGroup]) -> Result<(Vec<WeightedMoments>, Vec<usize>)> {
    let mut moments = Vec::with_capacity(groups.len() + 1);
    let mut samples = Vec::with_capacity(groups.len() + 1);
    moments.push(WeightedMoments::default());
    samples.push(0usize);
    for group in groups {
        moments.push(
            moments
                .last()
                .copied()
                .expect("prefix seed")
                .merge(group.moments)?,
        );
        samples.push(checked_sample_add(
            *samples.last().expect("prefix sample seed"),
            group.samples,
            "regression sample count",
        )?);
    }
    Ok((moments, samples))
}

fn regression_suffix(groups: &[RegressionGroup]) -> Result<(Vec<WeightedMoments>, Vec<usize>)> {
    let mut moments = vec![WeightedMoments::default(); groups.len() + 1];
    let mut samples = vec![0usize; groups.len() + 1];
    for index in (0..groups.len()).rev() {
        moments[index] = groups[index].moments.merge(moments[index + 1])?;
        samples[index] = checked_sample_add(
            samples[index + 1],
            groups[index].samples,
            "regression sample count",
        )?;
    }
    Ok((moments, samples))
}

fn checked_sample_add(left: usize, right: usize, operation: &'static str) -> Result<usize> {
    left.checked_add(right)
        .ok_or(Error::NumericalOverflow { operation })
}

fn add_counts(destination: &mut [CompensatedSum], source: &[CompensatedSum]) -> Result<()> {
    for (left, &right) in destination.iter_mut().zip(source) {
        left.merge(right, "class histogram merge")?;
    }
    Ok(())
}

fn totals(values: &[CompensatedSum], operation: &'static str) -> Result<Vec<f64>> {
    values.iter().map(|value| value.total(operation)).collect()
}

fn sum(values: &[f64], operation: &'static str) -> Result<f64> {
    let mut sum = CompensatedSum::default();
    for &value in values {
        sum.add(value, operation)?;
    }
    sum.total(operation)
}

#[allow(clippy::float_cmp)]
fn same_float(left: f64, right: f64) -> bool {
    // Exact identity is required here: values in the same group must route to
    // the same side of a threshold. Approximate equality would change CART.
    left == right
}
