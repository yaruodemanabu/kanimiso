//! Numerical regression values generated with Python POT 0.9.7.

use faer::Mat;
use wormhole::barycenter::{barycenter_with_options, BarycenterOptions};
use wormhole::coot::{co_optimal_transport, CootOptions, CootProblem};
use wormhole::factored::{factored_optimal_transport, FactoredOptions};
use wormhole::gaussian::{
    bures_distance, bures_wasserstein_distance, empirical_gaussian_gromov_wasserstein_distance,
    gaussian_gromov_wasserstein_distance, gaussian_gromov_wasserstein_mapping,
};
use wormhole::gmm::{
    apply_barycentric_map, component_cost, gaussian_log_pdf, gmm_barycenter_fixed_point,
    gmm_ot_loss, gmm_ot_plan, gmm_ot_plan_density, GaussianMixture, GmmBarycenterOptions,
};
use wormhole::gromov::{
    bapg_fused_gromov_wasserstein, fused_gromov_barycenter, gromov_barycenter, gromov_wasserstein,
    partial_fused_gromov_wasserstein, partial_gromov_wasserstein, BapgOptions,
    FusedGromovBarycenterOptions, FusedGromovBarycenterProblem, GromovBarycenterOptions,
    GromovOptions, PartialGromovMethod, PartialGromovOptions,
};
use wormhole::lowrank::{
    sinkhorn_low_rank_kernel, squared_euclidean_cost_factors, LowRankKernelOptions,
};
use wormhole::optim::{squared_l2_transport, ConditionalGradientOptions};
use wormhole::partial::{
    entropic_partial_wasserstein, partial_wasserstein_lagrange, EntropicPartialOptions,
};
use wormhole::sinkhorn::{
    empirical_sinkhorn_divergence, greenkhorn_with_options, sinkhorn_epsilon_scaling,
    sinkhorn_stabilized, sinkhorn_with_options, EpsilonScalingOptions, GreenkhornOptions,
    SinkhornOptions,
};
use wormhole::sliced::{
    expected_sliced_plan_with_projections, min_sliced_transport_plan_with_projections,
    sliced_plans_with_projections, sliced_wasserstein_with_projections, ExpectedSlicedPlanOptions,
};
use wormhole::unbalanced::{
    barycenter_unbalanced_with_options, sinkhorn_unbalanced_with_options,
    UnbalancedBarycenterOptions, UnbalancedOptions,
};
use wormhole::utils::{
    bures_exponential, normalize_cost, project_psd, project_simplex, project_sparse_simplex,
    CostNormalization,
};
use wormhole::weak::{weak_optimal_transport_with_options, WeakTransportOptions};
use wormhole::{emd, pairwise_batch, partial, BatchMetric, Metric, SolverStatus};

fn cost() -> Mat<f64> {
    Mat::<f64>::from_fn(3, 3, |i, j| {
        [[0.0, 1.0, 4.0], [1.0, 0.0, 1.0], [4.0, 1.0, 0.0]][i][j]
    })
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual:.16e}, expected={expected:.16e}, tolerance={tolerance:.2e}"
    );
}

fn assert_matrix_close(actual: &Mat<f64>, expected: &[&[f64]], tolerance: f64) {
    assert_eq!(actual.nrows(), expected.len());
    assert_eq!(actual.ncols(), expected[0].len());
    for (i, row) in expected.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            assert_close(actual[(i, j)], value, tolerance);
        }
    }
}

#[test]
fn exact_and_partial_match_pot_097() {
    let source = [0.2, 0.5, 0.3];
    let target = [0.4, 0.1, 0.5];
    let cost = cost();
    let exact = emd(&source, &target, &cost).unwrap();
    assert_close(exact.value, 0.4, 1e-12);
    assert_matrix_close(
        &exact.plan,
        &[&[0.2, 0.0, 0.0], &[0.2, 0.1, 0.2], &[0.0, 0.0, 0.3]],
        1e-12,
    );

    let partial = partial::partial_wasserstein(&source, &target, &cost, 0.6).unwrap();
    assert_close(partial.value, 0.0, 1e-12);
    assert_matrix_close(
        &partial.plan,
        &[&[0.2, 0.0, 0.0], &[0.0, 0.1, 0.0], &[0.0, 0.0, 0.3]],
        1e-12,
    );

    let lagrange_cost = Mat::<f64>::from_fn(2, 2, |i, j| [[0.0, 1.0], [2.0, 3.0]][i][j]);
    let lagrange =
        partial_wasserstein_lagrange(&[0.1, 0.2], &[0.1, 0.1], &lagrange_cost, Some(2.0)).unwrap();
    assert_matrix_close(&lagrange.plan, &[&[0.1, 0.0], &[0.0, 0.0]], 1e-12);
}

#[test]
fn sinkhorn_matches_pot_097_log_solver() {
    let source = [0.2, 0.5, 0.3];
    let target = [0.4, 0.1, 0.5];
    let result = sinkhorn_with_options(
        &source,
        &target,
        &cost(),
        0.7,
        SinkhornOptions {
            max_iterations: 10_000,
            tolerance: 1e-13,
            check_interval: 1,
            ..SinkhornOptions::default()
        },
    )
    .unwrap();
    assert_eq!(result.status, SolverStatus::Converged);
    assert_close(result.value, 0.4302617682972145, 1e-10);
    assert_matrix_close(
        &result.plan,
        &[
            &[
                0.1945378340950183,
                0.004811367081571924,
                0.0006507988234098154,
            ],
            &[
                0.20451271433636312,
                0.08806961663544934,
                0.20741766902818756,
            ],
            &[
                0.0009494515686184766,
                0.007119016282978768,
                0.29193153214840273,
            ],
        ],
        1e-10,
    );
}

#[test]
fn stabilized_and_epsilon_scaling_sinkhorn_match_pot_097() {
    let source = [0.2, 0.5, 0.3];
    let target = [0.4, 0.1, 0.5];
    let options = SinkhornOptions {
        max_iterations: 10_000,
        tolerance: 1e-12,
        check_interval: 1,
        ..SinkhornOptions::default()
    };
    let stabilized = sinkhorn_stabilized(&source, &target, &cost(), 0.1, options).unwrap();
    let scaled = sinkhorn_epsilon_scaling(
        &source,
        &target,
        &cost(),
        0.1,
        EpsilonScalingOptions {
            stages: 8,
            sinkhorn: options,
            ..EpsilonScalingOptions::default()
        },
    )
    .unwrap();
    let expected = &[
        &[
            0.19999999979388466,
            2.0611536075693164e-10,
            8.496_708_506_204_862e-19,
        ][..],
        &[
            0.20000000020611544,
            0.099_999_999_484_711_64,
            0.20000000030917284,
        ][..],
        &[
            1.2745062746172636e-18,
            3.0917304097608483e-10,
            0.29999999969082697,
        ][..],
    ];
    assert_matrix_close(&stabilized.plan, expected, 1e-10);
    assert_matrix_close(&scaled.plan, expected, 1e-10);
    assert_close(scaled.value, 0.40000000103057687, 1e-10);
}

#[test]
fn greenkhorn_matches_pot_097() {
    let source = [0.2, 0.5, 0.3];
    let target = [0.4, 0.1, 0.5];
    let result = greenkhorn_with_options(
        &source,
        &target,
        &cost(),
        0.7,
        GreenkhornOptions {
            max_iterations: 100_000,
            tolerance: 1e-13,
        },
    )
    .unwrap();
    assert_eq!(result.status, SolverStatus::Converged);
    assert_close(result.value, 0.43026176829718443, 1e-10);
    assert_matrix_close(
        &result.plan,
        &[
            &[
                0.19453783409501632,
                0.00481136708157346,
                0.00065079882341025,
            ],
            &[
                0.20451271433628018,
                0.08806961663544248,
                0.20741766902824438,
            ],
            &[
                0.00094945156861764,
                0.00711901628297486,
                0.29193153214834505,
            ],
        ],
        1e-10,
    );
}

#[test]
fn empirical_sinkhorn_divergence_matches_pot_097_linear_cost() {
    let source = Mat::<f64>::from_fn(2, 1, |i, _| i as f64);
    let target = Mat::<f64>::from_fn(4, 1, |i, _| i as f64);
    let value =
        empirical_sinkhorn_divergence(&source, &target, None, None, Metric::SquaredEuclidean, 0.5)
            .unwrap();
    assert_close(value, 1.4203817720317107, 1e-8);
}

#[test]
fn unbalanced_sinkhorn_matches_pot_097() {
    let source = [0.4, 0.8];
    let target = [0.3, 0.2];
    let cost = Mat::<f64>::from_fn(2, 2, |i, j| [[0.0, 2.0], [1.0, 0.0]][i][j]);
    let result = sinkhorn_unbalanced_with_options(
        &source,
        &target,
        &cost,
        0.5,
        UnbalancedOptions {
            source_penalty: 1.0,
            target_penalty: 1.0,
            max_iterations: 10_000,
            tolerance: 1e-13,
            ..UnbalancedOptions::default()
        },
    )
    .unwrap();
    assert_eq!(result.status, SolverStatus::Converged);
    assert_matrix_close(
        &result.plan,
        &[
            &[0.2432515016490088, 0.0017683610608789104],
            &[0.10073611769636161, 0.2954387887494088],
        ],
        1e-9,
    );
}

#[test]
fn unbalanced_barycenter_matches_pot_097() {
    let distributions =
        Mat::<f64>::from_fn(3, 2, |i, j| [[0.2, 0.4], [0.5, 0.1], [0.3, 0.8]][i][j]);
    let result = barycenter_unbalanced_with_options(
        &distributions,
        &cost(),
        0.7,
        1.2,
        Some(&[0.25, 0.75]),
        UnbalancedBarycenterOptions {
            max_iterations: 10_000,
            tolerance: 1e-13,
        },
    )
    .unwrap();
    assert_eq!(result.status, SolverStatus::Converged);
    for (&actual, expected) in
        result
            .weights
            .iter()
            .zip([0.5100783933091891, 0.5290423579381602, 0.730534186510739])
    {
        assert_close(actual, expected, 1e-10);
    }
}

#[test]
fn entropic_partial_matches_pot_097() {
    let source = [0.2, 0.5, 0.3];
    let target = [0.4, 0.1, 0.5];
    let result = entropic_partial_wasserstein(
        &source,
        &target,
        &cost(),
        0.7,
        0.6,
        EntropicPartialOptions {
            max_iterations: 10_000,
            tolerance: 1e-13,
        },
    )
    .unwrap();
    assert_eq!(result.status, SolverStatus::Converged);
    assert_matrix_close(
        &result.plan,
        &[
            &[
                0.18496214174810457,
                0.014427759562708534,
                0.0006100986891869455,
            ],
            &[0.050824840857958, 0.06902929769890181, 0.050824840857958],
            &[
                0.00069954227031016,
                0.016542942738389684,
                0.21207853557648226,
            ],
        ],
        1e-9,
    );
}

#[test]
fn fixed_barycenter_matches_pot_097() {
    let distributions =
        Mat::<f64>::from_fn(3, 2, |i, j| [[0.2, 0.4], [0.5, 0.1], [0.3, 0.5]][i][j]);
    let result = barycenter_with_options(
        &distributions,
        &cost(),
        0.7,
        None,
        BarycenterOptions {
            max_iterations: 10_000,
            tolerance: 1e-13,
        },
    )
    .unwrap();
    for (&actual, expected) in
        result
            .weights
            .iter()
            .zip([0.292474390164342, 0.3373529776715748, 0.3701726321640831])
    {
        assert_close(actual, expected, 1e-9);
    }
}

#[test]
fn sliced_gaussian_and_gromov_values_match_pot_097() {
    let source = Mat::<f64>::from_fn(3, 2, |i, j| [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]][i][j]);
    let target = Mat::<f64>::from_fn(2, 2, |i, j| [[1.0, 1.0], [2.0, 1.0]][i][j]);
    let inverse_root_two = 2.0_f64.sqrt().recip();
    let projections = Mat::<f64>::from_fn(2, 3, |i, j| {
        [[1.0, 0.0, inverse_root_two], [0.0, 1.0, inverse_root_two]][i][j]
    });
    let sliced = sliced_wasserstein_with_projections(
        &source,
        &target,
        &[1.0 / 3.0; 3],
        &[0.5; 2],
        &projections,
        2.0,
    )
    .unwrap();
    assert_close(sliced, 1.1426091000668408, 1e-12);

    let first = Mat::<f64>::from_fn(2, 2, |i, j| [[4.0, 1.0], [1.0, 2.0]][i][j]);
    let second = Mat::<f64>::from_fn(2, 2, |i, j| [[3.0, 0.2], [0.2, 1.0]][i][j]);
    assert_close(
        bures_distance(&first, &second).unwrap(),
        0.5512232617114855,
        1e-10,
    );
    assert_close(
        bures_wasserstein_distance(&[0.0, 1.0], &[2.0, -1.0], &first, &second).unwrap(),
        2.881639652047398,
        1e-10,
    );

    let source_structure = Mat::<f64>::from_fn(2, 2, |i, j| [[0.0, 1.0], [1.0, 0.0]][i][j]);
    let target_structure = Mat::<f64>::from_fn(2, 2, |i, j| [[0.0, 2.0], [2.0, 0.0]][i][j]);
    let gw = gromov_wasserstein(
        &source_structure,
        &target_structure,
        &[0.5, 0.5],
        &[0.5, 0.5],
        GromovOptions::default(),
    )
    .unwrap();
    assert_close(gw.value, 0.5, 1e-10);
}

#[test]
fn gaussian_gromov_quantities_match_pot_097() {
    let source_covariance = Mat::<f64>::from_fn(2, 2, |i, j| [[4.0, 1.0], [1.0, 1.0]][i][j]);
    let target_covariance = Mat::<f64>::from_fn(1, 1, |_, _| 9.0);
    assert_close(
        gaussian_gromov_wasserstein_distance(&source_covariance, &target_covariance).unwrap(),
        15.633307652783936,
        1e-12,
    );
    let mapping = gaussian_gromov_wasserstein_mapping(
        &[1.0, -1.0],
        &[2.0],
        &source_covariance,
        &target_covariance,
        None,
    )
    .unwrap();
    assert_close(mapping.apply(&[1.0, -1.0]).unwrap()[0], 2.0, 1e-12);
    let first = &mapping.linear * &source_covariance;
    let mapped_covariance = &first * mapping.linear.transpose();
    assert_close(mapped_covariance[(0, 0)], 9.0, 1e-10);

    let source_samples = Mat::<f64>::from_fn(4, 2, |i, j| {
        [[0.0, 0.0], [2.0, 0.0], [0.0, 3.0], [1.0, -1.0]][i][j]
    });
    let target_samples = Mat::<f64>::from_fn(3, 1, |i, _| [0.0, 2.0, 4.0][i]);
    assert_close(
        empirical_gaussian_gromov_wasserstein_distance(
            &source_samples,
            &target_samples,
            None,
            None,
        )
        .unwrap(),
        1.5360273707785967,
        1e-12,
    );
}

#[test]
fn minimum_sliced_plan_matches_pot_097_original_space_cost() {
    let source = Mat::<f64>::from_fn(2, 2, |i, j| [[3.0, 3.0], [1.0, 1.0]][i][j]);
    let target = Mat::<f64>::from_fn(2, 2, |i, j| [[2.0, 2.5], [3.0, 2.0]][i][j]);
    let projections = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });
    let result =
        min_sliced_transport_plan_with_projections(&source, &target, None, None, &projections, 2.0)
            .unwrap();
    assert_close(result.value, 2.125, 1e-12);
    assert_matrix_close(&result.plan, &[&[0.0, 0.5], &[0.5, 0.0]], 1e-12);

    let candidates = sliced_plans_with_projections(
        &source,
        &target,
        None,
        None,
        &projections,
        Metric::SquaredEuclidean,
        2.0,
    )
    .unwrap();
    assert_eq!(candidates.costs, vec![2.125, 3.125]);

    let expected = expected_sliced_plan_with_projections(
        &source,
        &target,
        None,
        None,
        &projections,
        ExpectedSlicedPlanOptions {
            beta: 1.5,
            ..ExpectedSlicedPlanOptions::default()
        },
    )
    .unwrap();
    assert_close(expected.value, 2.3074255238063563, 1e-12);
    assert_matrix_close(
        &expected.plan,
        &[
            &[0.09121276190317816, 0.4087872380968218],
            &[0.4087872380968218, 0.09121276190317816],
        ],
        1e-12,
    );
}

#[test]
fn gaussian_mixture_quantities_match_pot_097() {
    let source = GaussianMixture::new(
        Mat::<f64>::from_fn(2, 1, |i, _| [0.0, 3.0][i]),
        vec![
            Mat::<f64>::from_fn(1, 1, |_, _| 1.0),
            Mat::<f64>::from_fn(1, 1, |_, _| 4.0),
        ],
        vec![0.4, 0.6],
    )
    .unwrap();
    let target = GaussianMixture::new(
        Mat::<f64>::from_fn(2, 1, |i, _| [1.0, 5.0][i]),
        vec![
            Mat::<f64>::from_fn(1, 1, |_, _| 2.25),
            Mat::<f64>::from_fn(1, 1, |_, _| 1.0),
        ],
        vec![0.7, 0.3],
    )
    .unwrap();
    let samples = Mat::<f64>::from_fn(4, 1, |i, _| [0.0, 1.0, 3.0, 6.0][i]);

    let log_density =
        gaussian_log_pdf(&samples, &[0.0], &Mat::<f64>::from_fn(1, 1, |_, _| 1.0)).unwrap();
    for (&actual, expected) in log_density.iter().zip([
        -0.9189385332046727,
        -1.4189385332046727,
        -5.418938533204672,
        -18.918938533204674,
    ]) {
        assert_close(actual, expected, 1e-12);
    }

    for (&actual, expected) in source.pdf(&samples).unwrap().iter().zip([
        0.1984321908603406,
        0.16937950716340036,
        0.121455423485205,
        0.03885528113012066,
    ]) {
        assert_close(actual, expected, 1e-12);
    }

    assert_matrix_close(
        &component_cost(&source, &target).unwrap(),
        &[&[1.25, 25.0], &[4.25, 5.0]],
        1e-12,
    );
    let plan = gmm_ot_plan(&source, &target).unwrap();
    assert_matrix_close(&plan.plan, &[&[0.4, 0.0], &[0.3, 0.3]], 1e-12);
    assert_close(gmm_ot_loss(&source, &target).unwrap(), 3.275, 1e-12);

    let mapped = apply_barycentric_map(&samples, &source, &target, Some(&plan.plan)).unwrap();
    assert_matrix_close(
        &mapped,
        &[
            &[1.0244764209698682],
            &[2.178571428571429],
            &[3.036489506065391],
            &[4.8750003205628545],
        ],
        1e-12,
    );

    let density_source = GaussianMixture::new(
        Mat::<f64>::zeros(1, 1),
        vec![Mat::<f64>::from_fn(1, 1, |_, _| 1.0)],
        vec![1.0],
    )
    .unwrap();
    let density_target = GaussianMixture::new(
        Mat::<f64>::from_fn(1, 1, |_, _| 1.0),
        vec![Mat::<f64>::from_fn(1, 1, |_, _| 4.0)],
        vec![1.0],
    )
    .unwrap();
    let source_points = Mat::<f64>::from_fn(3, 1, |i, _| [-1.0, 0.0, 1.0][i]);
    let target_points = Mat::<f64>::from_fn(3, 1, |i, _| [-1.0, 1.0, 3.0][i]);
    assert_matrix_close(
        &gmm_ot_plan_density(
            &source_points,
            &target_points,
            &density_source,
            &density_target,
            None,
            1e-2,
        )
        .unwrap(),
        &[
            &[0.24197072451914337, 0.0, 0.0],
            &[0.0, 0.3989422804014327, 0.0],
            &[0.0, 0.0, 0.24197072451914337],
        ],
        1e-12,
    );
}

#[test]
fn gaussian_mixture_barycenter_matches_pot_097() {
    let first = GaussianMixture::new(
        Mat::<f64>::from_fn(2, 1, |i, _| [0.0, 4.0][i]),
        vec![
            Mat::<f64>::from_fn(1, 1, |_, _| 1.0),
            Mat::<f64>::from_fn(1, 1, |_, _| 4.0),
        ],
        vec![0.5, 0.5],
    )
    .unwrap();
    let second = GaussianMixture::new(
        Mat::<f64>::from_fn(2, 1, |i, _| [2.0, 6.0][i]),
        vec![
            Mat::<f64>::from_fn(1, 1, |_, _| 9.0),
            Mat::<f64>::from_fn(1, 1, |_, _| 1.0),
        ],
        vec![0.5, 0.5],
    )
    .unwrap();
    let initial = GaussianMixture::new(
        Mat::<f64>::from_fn(2, 1, |i, _| [1.0, 5.0][i]),
        vec![
            Mat::<f64>::from_fn(1, 1, |_, _| 2.0),
            Mat::<f64>::from_fn(1, 1, |_, _| 2.0),
        ],
        vec![0.5, 0.5],
    )
    .unwrap();
    let barycenter = gmm_barycenter_fixed_point(
        &[first, second],
        &initial,
        Some(&[0.25, 0.75]),
        GmmBarycenterOptions {
            iterations: 2,
            ..GmmBarycenterOptions::default()
        },
    )
    .unwrap();
    assert_matrix_close(&barycenter.mixture.means, &[&[1.5], &[5.5]], 1e-12);
    assert_close(
        barycenter.mixture.covariances[0][(0, 0)],
        6.249999892000391,
        2e-7,
    );
    assert_close(
        barycenter.mixture.covariances[1][(0, 0)],
        1.562499892000393,
        2e-7,
    );
}

#[test]
fn factored_transport_matches_pot_097() {
    let source = Mat::<f64>::from_fn(3, 1, |i, _| [0.0, 2.0, 6.0][i]);
    let target = Mat::<f64>::from_fn(2, 1, |i, _| [1.0, 4.0][i]);
    let initial_support = Mat::<f64>::from_fn(2, 1, |i, _| [0.0, 5.0][i]);
    let result = factored_optimal_transport(
        &source,
        &target,
        Some(&[0.2, 0.5, 0.3]),
        Some(&[0.6, 0.4]),
        Some(&initial_support),
        FactoredOptions {
            rank: 2,
            max_iterations: 100,
            tolerance: 1e-12,
            ..FactoredOptions::default()
        },
    )
    .unwrap();

    assert_eq!(result.status, SolverStatus::Converged);
    assert_matrix_close(
        &result.source_plan,
        &[&[0.2, 0.0], &[0.3, 0.2], &[0.0, 0.3]],
        1e-12,
    );
    assert_matrix_close(&result.target_plan, &[&[0.5, 0.0], &[0.1, 0.4]], 1e-12);
    assert_matrix_close(&result.support, &[&[1.1], &[3.9]], 1e-12);
    assert_close(result.value, 3.38, 1e-12);
}

#[test]
fn batched_distances_match_pot_097() {
    let left = vec![
        Mat::<f64>::from_fn(2, 2, |i, j| [[0.0, 0.0], [1.0, 0.0]][i][j]),
        Mat::<f64>::from_fn(2, 2, |i, j| [[0.0, 1.0], [2.0, 1.0]][i][j]),
    ];
    let right = vec![
        Mat::<f64>::from_fn(2, 2, |i, j| [[1.0, 1.0], [2.0, 0.0]][i][j]),
        Mat::<f64>::from_fn(2, 2, |i, j| [[1.0, 0.0], [3.0, 2.0]][i][j]),
    ];
    let squared = pairwise_batch(
        &left,
        Some(&right),
        BatchMetric::Distance(Metric::SquaredEuclidean),
    )
    .unwrap();
    assert_matrix_close(&squared[0], &[&[2.0, 4.0], &[1.0, 1.0]], 1e-12);
    assert_matrix_close(&squared[1], &[&[2.0, 10.0], &[2.0, 2.0]], 1e-12);

    let probabilities = vec![
        Mat::<f64>::from_fn(2, 2, |i, j| [[0.2, 0.8], [0.5, 0.5]][i][j]),
        Mat::<f64>::from_fn(2, 2, |i, j| [[0.1, 0.9], [0.25, 0.75]][i][j]),
    ];
    let targets = vec![
        Mat::<f64>::from_fn(1, 2, |_, j| [0.4, 0.6][j]),
        Mat::<f64>::from_fn(1, 2, |_, j| [0.8, 0.2][j]),
    ];
    let kl = pairwise_batch(&probabilities, Some(&targets), BatchMetric::KullbackLeibler).unwrap();
    assert_matrix_close(
        &kl[0],
        &[&[0.1046496286779095], &[0.02013551355068877]],
        1e-12,
    );
    assert_matrix_close(
        &kl[1],
        &[&[1.362737753366392], &[0.6661694797014142]],
        1e-12,
    );
}

#[test]
fn support_utilities_match_pot_097() {
    let values = [-0.5, 0.3, 1.2];
    for (&actual, expected) in project_simplex(&values, 1.0)
        .unwrap()
        .iter()
        .zip([0.0, 0.05, 0.95])
    {
        assert_close(actual, expected, 1e-12);
    }
    assert_eq!(
        project_sparse_simplex(&values, 1.0, 1).unwrap(),
        vec![0.0, 0.0, 1.0]
    );

    let cost = Mat::<f64>::from_fn(2, 2, |i, j| [[0.0, 1.0], [3.0, 7.0]][i][j]);
    assert_matrix_close(
        &normalize_cost(&cost, CostNormalization::Median).unwrap(),
        &[&[0.0, 0.5], &[1.5, 3.5]],
        1e-12,
    );
    assert_matrix_close(
        &normalize_cost(&cost, CostNormalization::LogLog).unwrap(),
        &[
            &[0.0, 0.5265890341390446],
            &[0.8697416861919439, 1.1247482629090362],
        ],
        1e-12,
    );

    let indefinite = Mat::<f64>::from_fn(2, 2, |i, j| [[1.0, 2.0], [2.0, 1.0]][i][j]);
    assert_matrix_close(
        &project_psd(&indefinite, 0.0).unwrap(),
        &[&[1.5, 1.5], &[1.5, 1.5]],
        1e-12,
    );
    let covariance = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { [4.0, 9.0][i] } else { 0.0 });
    let tangent = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { [0.5, -0.5][i] } else { 0.0 });
    assert_matrix_close(
        &bures_exponential(&covariance, &tangent).unwrap(),
        &[&[9.0, 0.0], &[0.0, 2.25]],
        1e-12,
    );
}

#[test]
fn low_rank_kernel_quantities_match_pot_097() {
    let source = Mat::<f64>::from_fn(3, 2, |i, j| [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]][i][j]);
    let target = Mat::<f64>::from_fn(2, 2, |i, j| [[1.0, 1.0], [2.0, 0.0]][i][j]);
    let cost_factors = squared_euclidean_cost_factors(&source, &target, false).unwrap();
    assert_matrix_close(
        &cost_factors.dense(),
        &[&[2.0, 4.0], &[1.0, 1.0], &[2.0, 8.0]],
        1e-12,
    );

    let kernel_left = Mat::<f64>::from_fn(3, 2, |i, j| [[1.0, 0.2], [0.5, 1.0], [0.2, 0.7]][i][j]);
    let kernel_right = Mat::<f64>::from_fn(2, 2, |i, j| [[0.8, 0.3], [0.4, 1.1]][i][j]);
    let result = sinkhorn_low_rank_kernel(
        &kernel_left,
        &kernel_right,
        Some(&[0.2, 0.5, 0.3]),
        Some(&[0.6, 0.4]),
        LowRankKernelOptions {
            max_iterations: 10_000,
            tolerance: 1e-13,
            ..LowRankKernelOptions::default()
        },
    )
    .unwrap();
    assert_matrix_close(
        &result.dense_plan(),
        &[
            &[0.1554654248237166, 0.0445345751762834],
            &[0.2876984705011858, 0.21230152949881423],
            &[0.15683610467509745, 0.1431638953249025],
        ],
        1e-12,
    );
}

#[test]
fn weak_transport_matches_pot_097() {
    let source = Mat::<f64>::from_fn(3, 1, |i, _| [0.0, 2.0, 5.0][i]);
    let target = Mat::<f64>::from_fn(2, 1, |i, _| [1.0, 4.0][i]);
    let result = weak_optimal_transport_with_options(
        &source,
        &target,
        Some(&[0.2, 0.5, 0.3]),
        Some(&[0.6, 0.4]),
        None,
        WeakTransportOptions {
            max_iterations: 1_000,
            relative_tolerance: 1e-12,
            absolute_tolerance: 1e-12,
        },
    )
    .unwrap();
    assert_close(result.value, 0.58, 1e-10);
    assert_matrix_close(
        &result.plan,
        &[&[0.2, 0.0], &[0.4, 0.1], &[0.0, 0.3]],
        1e-10,
    );
}

#[test]
fn conditional_gradient_squared_l2_matches_pot_097() {
    let source = [0.2, 0.5, 0.3];
    let target = [0.4, 0.1, 0.5];
    let result = squared_l2_transport(
        &source,
        &target,
        &cost(),
        2.0,
        ConditionalGradientOptions {
            max_iterations: 1_000,
            gap_tolerance: 1e-12,
            objective_tolerance: 1e-12,
            ..ConditionalGradientOptions::default()
        },
    )
    .unwrap();
    assert_close(result.objective, 0.6200000000000001, 1e-10);
    assert_matrix_close(
        &result.plan,
        &[&[0.2, 0.0, 0.0], &[0.2, 0.1, 0.2], &[0.0, 0.0, 0.3]],
        1e-10,
    );
}

#[test]
fn balanced_coot_matches_pot_097() {
    let source = Mat::<f64>::from_fn(3, 2, |i, j| [[0.0, 1.0], [2.0, 3.0], [4.0, 0.0]][i][j]);
    let target = Mat::<f64>::from_fn(2, 3, |i, j| [[1.0, 2.0, 0.0], [3.0, 0.0, 4.0]][i][j]);
    let result = co_optimal_transport(
        CootProblem {
            source: &source,
            target: &target,
            source_sample_weights: Some(&[0.2, 0.5, 0.3]),
            source_feature_weights: Some(&[0.7, 0.3]),
            target_sample_weights: Some(&[0.6, 0.4]),
            target_feature_weights: Some(&[0.2, 0.5, 0.3]),
            sample_linear_cost: None,
            feature_linear_cost: None,
        },
        CootOptions {
            max_iterations: 1_000,
            tolerance: 1e-12,
            objective_tolerance: 1e-12,
            ..CootOptions::default()
        },
    )
    .unwrap();
    assert_close(result.value, 2.4499999999999993, 1e-12);
    assert_matrix_close(
        &result.sample_plan,
        &[&[0.2, 0.0], &[0.4, 0.1], &[0.0, 0.3]],
        1e-12,
    );
    assert_matrix_close(
        &result.feature_plan,
        &[&[0.2, 0.2, 0.3], &[0.0, 0.3, 0.0]],
        1e-12,
    );
}

#[test]
fn gromov_barycenter_matches_pot_097() {
    let first = Mat::<f64>::from_fn(2, 2, |i, j| i.abs_diff(j) as f64);
    let second = Mat::<f64>::from_fn(3, 3, |i, j| {
        [[0.0, 2.0, 3.0], [2.0, 0.0, 1.0], [3.0, 1.0, 0.0]][i][j]
    });
    let initial = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 0.0 } else { 1.5 });
    let result = gromov_barycenter(
        &[first, second],
        Some(&[vec![0.5, 0.5], vec![0.2, 0.5, 0.3]]),
        &[0.4, 0.6],
        Some(&[0.25, 0.75]),
        &initial,
        GromovBarycenterOptions {
            max_iterations: 1_000,
            tolerance: 1e-12,
            ..GromovBarycenterOptions::default()
        },
    )
    .unwrap();
    assert_matrix_close(
        &result.structure,
        &[
            &[1.1250000000000002, 1.3333333333333335],
            &[1.3333333333333335, 0.27777777777777773],
        ],
        1e-12,
    );
    assert_matrix_close(&result.plans[0], &[&[0.0, 0.4], &[0.5, 0.1]], 1e-12);
    assert_matrix_close(
        &result.plans[1],
        &[&[0.2, 0.0, 0.2], &[0.0, 0.5, 0.1]],
        1e-12,
    );
}

#[test]
fn fused_gromov_barycenter_matches_pot_097() {
    let first_features = Mat::<f64>::from_fn(2, 1, |i, _| [0.0, 2.0][i]);
    let first_structure = Mat::<f64>::from_fn(2, 2, |i, j| i.abs_diff(j) as f64);
    let second_features = Mat::<f64>::from_fn(3, 1, |i, _| [1.0, 3.0, 5.0][i]);
    let second_structure = Mat::<f64>::from_fn(3, 3, |i, j| {
        [[0.0, 2.0, 3.0], [2.0, 0.0, 1.0], [3.0, 1.0, 0.0]][i][j]
    });
    let initial_features = Mat::<f64>::from_fn(2, 1, |i, _| [0.5, 3.0][i]);
    let initial_structure = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 0.0 } else { 1.5 });
    let result = fused_gromov_barycenter(
        FusedGromovBarycenterProblem {
            features: &[first_features, second_features],
            structures: &[first_structure, second_structure],
            distributions: Some(&[vec![0.5, 0.5], vec![0.2, 0.5, 0.3]]),
            barycenter_weights: &[0.4, 0.6],
            mixture: Some(&[0.25, 0.75]),
            initial_features: &initial_features,
            initial_structure: &initial_structure,
        },
        FusedGromovBarycenterOptions {
            structure_weight: 0.6,
            max_iterations: 1_000,
            tolerance: 1e-12,
            ..FusedGromovBarycenterOptions::default()
        },
    )
    .unwrap();
    assert_matrix_close(
        &result.features,
        &[&[1.5000000000000002], &[3.4166666666666665]],
        1e-12,
    );
    assert_matrix_close(
        &result.structure,
        &[
            &[0.7500000000000001, 1.3333333333333335],
            &[1.3333333333333335, 0.44444444444444453],
        ],
        1e-12,
    );
    assert_matrix_close(&result.plans[0], &[&[0.4, 0.0], &[0.1, 0.5]], 1e-12);
    assert_matrix_close(
        &result.plans[1],
        &[&[0.2, 0.2, 0.0], &[0.0, 0.3, 0.3]],
        1e-12,
    );
}

#[test]
fn bapg_fused_gromov_matches_pot_097() {
    let source_structure = Mat::<f64>::from_fn(3, 3, |i, j| {
        [[0.0, 1.0, 3.0], [1.0, 0.0, 2.0], [3.0, 2.0, 0.0]][i][j]
    });
    let target_structure = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 0.0 } else { 2.0 });
    let feature_cost = Mat::<f64>::from_fn(3, 2, |i, j| [[0.0, 1.0], [1.0, 0.0], [2.0, 0.5]][i][j]);
    let result = bapg_fused_gromov_wasserstein(
        &feature_cost,
        &source_structure,
        &target_structure,
        &[0.2, 0.5, 0.3],
        &[0.6, 0.4],
        0.6,
        BapgOptions {
            regularization: 0.5,
            max_iterations: 10_000,
            tolerance: 1e-12,
            marginal_loss: true,
        },
    )
    .unwrap();
    assert_close(result.value, 0.4544104167895048, 1e-10);
    assert_matrix_close(
        &result.plan,
        &[
            &[0.10450772839556065, 0.0],
            &[0.49549227160443937, 0.0],
            &[0.0, 0.4],
        ],
        1e-10,
    );
}

#[test]
fn partial_gromov_solvers_match_pot_097() {
    let source_structure = Mat::<f64>::from_fn(3, 3, |i, j| {
        [[0.0, 1.0, 3.0], [1.0, 0.0, 2.0], [3.0, 2.0, 0.0]][i][j]
    });
    let target_structure = Mat::<f64>::from_fn(2, 2, |i, j| if i == j { 0.0 } else { 2.0 });
    let feature_cost = Mat::<f64>::from_fn(3, 2, |i, j| [[0.0, 1.0], [1.0, 0.0], [2.0, 0.5]][i][j]);
    let source = [0.2, 0.5, 0.3];
    let target = [0.6, 0.4];
    let exact = partial_fused_gromov_wasserstein(
        &feature_cost,
        &source_structure,
        &target_structure,
        &source,
        &target,
        0.6,
        PartialGromovOptions {
            transported_mass: 0.7,
            tolerance: 1e-12,
            ..PartialGromovOptions::default()
        },
    )
    .unwrap();
    assert_close(exact.value, 0.35199999999999965, 1e-10);
    assert_matrix_close(&exact.plan, &[&[0.2, 0.0], &[0.1, 0.4], &[0.0, 0.0]], 1e-10);

    let entropic = partial_gromov_wasserstein(
        &source_structure,
        &target_structure,
        &source,
        &target,
        PartialGromovOptions {
            transported_mass: 0.7,
            max_iterations: 10_000,
            tolerance: 1e-12,
            method: PartialGromovMethod::Entropic {
                regularization: 0.2,
                options: EntropicPartialOptions {
                    max_iterations: 10_000,
                    tolerance: 1e-12,
                },
            },
        },
    )
    .unwrap();
    assert_close(entropic.value, 0.20010907764740438, 1e-8);
    assert_matrix_close(
        &entropic.plan,
        &[
            &[6.743622454957472e-5, 0.19993209989861],
            &[0.49999999896944153, 1.030_558_452_388_142e-9],
            &[8.254994911957286e-9, 4.5562184551612335e-7],
        ],
        1e-7,
    );
}
