//! State-space filters: linear Kalman, nonlinear Kalman, and particle
//! filters, plus the usual extensions (smoothers, information / square-root
//! forms, EnKF, APF, RPF, FFBSi).
//!
//! All Gaussian filters share the discrete additive-noise model
//! [`DiscreteSsm`]. Linear SDEs still go through [`kalman_bucy`]
//! (YUIMA `KalmanBucy`) with the exact `exp(A Δt)` transition.

mod kalman;
mod nonlinear;
mod particle;
mod particle_model;
mod ssm;
mod ssm_est;

pub use kalman::{
    adaptive_kalman, information_filter, kalman, kalman_bucy, rts_smoother, square_root_kalman,
    AdaptiveKalmanConfig, KalmanBucy,
};
pub use nonlinear::{
    continuous_discrete_ekf, cubature_kalman, ensemble_kalman, extended_kalman,
    extended_rts_smoother, gaussian_sum_filter, iterated_ekf, second_order_ekf, unscented_kalman,
    EnkfConfig, IekfConfig, UkfParams,
};
pub use particle::{
    auxiliary_particle_filter, particle_filter, particle_smoother, regularized_particle_filter,
    sis_filter, unscented_particle_filter, ParticleConfig, ParticleFilter, RegularizedConfig,
    ResamplingScheme,
};
pub use particle_model::{
    conditional_smc, particle_filter_model, pmmh, smc2, GaussianParticle, ParticleModel, PmmhFit,
    PoissonObs, Smc2Fit, StudentTObs, TobitObs,
};
pub use ssm::{DiscreteSsm, FnSsm, GaussianFilter, GaussianSmoother, LinearGaussian, SdeSsm};
pub use ssm_est::{
    carma_as_linear_gaussian, diffuse_prior, shumway_stoffer_em, ssm_mle, LgParametrization,
    SsmEmFit,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::{col_from_slice, mat_from_row_slice};
    use crate::path::Path;
    use crate::rng::seed_rng;
    use amatsuki::{Rng, StandardNormal};

    fn linear_toy() -> (LinearGaussian, Path) {
        let model = LinearGaussian::new(
            mat_from_row_slice(1, 1, &[0.85]),
            mat_from_row_slice(1, 1, &[0.04]),
            mat_from_row_slice(1, 1, &[1.0]),
            mat_from_row_slice(1, 1, &[0.09]),
        )
        .unwrap();
        let n = 25;
        let times: Vec<f64> = (0..=n).map(|i| i as f64).collect();
        let mut vals = vec![0.0; n + 1];
        let mut x = 0.0;
        let mut rng = seed_rng(11);
        for i in 1..=n {
            x = 0.85 * x + 0.2 * rng.sample(StandardNormal);
            vals[i] = x + 0.3 * rng.sample(StandardNormal);
        }
        (model, Path::new(times, vals, 1).unwrap())
    }

    #[test]
    fn linear_gaussian_filters_agree() {
        let (model, obs) = linear_toy();
        let x0 = col_from_slice(&[0.0]);
        let p0 = mat_from_row_slice(1, 1, &[1.0]);
        let kf = kalman(&model, &obs, &x0, &p0).unwrap();
        let ekf = extended_kalman(&model, &obs, &x0, &p0).unwrap();
        let iekf = iterated_ekf(&model, &obs, &x0, &p0, IekfConfig::default()).unwrap();
        let so = second_order_ekf(&model, &obs, &x0, &p0).unwrap();
        let ukf = unscented_kalman(&model, &obs, &x0, &p0, UkfParams::default()).unwrap();
        let ckf = cubature_kalman(&model, &obs, &x0, &p0).unwrap();
        let inf = information_filter(&model, &obs, &x0, &p0).unwrap();
        for i in 0..kf.filtered.len() {
            let x = kf.filtered[i][0];
            assert!((x - ekf.filtered[i][0]).abs() < 1e-7, "ekf {i}");
            assert!((x - iekf.filtered[i][0]).abs() < 1e-7, "iekf {i}");
            assert!((x - so.filtered[i][0]).abs() < 2e-2, "soekf {i}");
            assert!((x - ukf.filtered[i][0]).abs() < 1e-7, "ukf {i}");
            assert!((x - ckf.filtered[i][0]).abs() < 1e-7, "ckf {i}");
            assert!((x - inf.filtered[i][0]).abs() < 1e-7, "inf {i}");
        }
        let sm = rts_smoother(&model, &kf).unwrap();
        let esm = extended_rts_smoother(&model, &obs, &ekf).unwrap();
        assert!((sm.smoothed.last().unwrap()[0] - kf.filtered.last().unwrap()[0]).abs() < 1e-12);
        assert!((esm.smoothed.last().unwrap()[0] - ekf.filtered.last().unwrap()[0]).abs() < 1e-12);
        let mix = gaussian_sum_filter(
            &model,
            &obs,
            &[
                (x0.clone(), p0.clone(), 0.5),
                (col_from_slice(&[0.2]), p0.clone(), 0.5),
            ],
        )
        .unwrap();
        assert_eq!(mix.filtered.len(), kf.filtered.len());
    }

    #[test]
    fn particle_and_ensemble_track_linear() {
        let (model, obs) = linear_toy();
        let x0 = col_from_slice(&[0.0]);
        let p0 = mat_from_row_slice(1, 1, &[1.0]);
        let kf = kalman(&model, &obs, &x0, &p0).unwrap();
        let mut rng = seed_rng(21);
        let pf = particle_filter(
            &model,
            &obs,
            &x0,
            &p0,
            ParticleConfig {
                n_particles: 400,
                ..ParticleConfig::default()
            },
            &mut rng,
        )
        .unwrap();
        let apf = auxiliary_particle_filter(
            &model,
            &obs,
            &x0,
            &p0,
            ParticleConfig {
                n_particles: 300,
                ..ParticleConfig::default()
            },
            &mut rng,
        )
        .unwrap();
        let rpf = regularized_particle_filter(
            &model,
            &obs,
            &x0,
            &p0,
            RegularizedConfig {
                particle: ParticleConfig {
                    n_particles: 250,
                    ..ParticleConfig::default()
                },
                ..RegularizedConfig::default()
            },
            &mut rng,
        )
        .unwrap();
        let enkf = ensemble_kalman(
            &model,
            &obs,
            &x0,
            &p0,
            EnkfConfig {
                n_ensemble: 80,
                inflation: 1.0,
            },
            &mut rng,
        )
        .unwrap();
        let sis = sis_filter(&model, &obs, &x0, &p0, 200, &mut rng).unwrap();
        let upf = unscented_particle_filter(
            &model,
            &obs,
            &x0,
            &p0,
            ParticleConfig {
                n_particles: 80,
                ..ParticleConfig::default()
            },
            UkfParams::default(),
            &mut rng,
        )
        .unwrap();
        let sm = particle_smoother(
            &model,
            &obs,
            &x0,
            &p0,
            ParticleConfig {
                n_particles: 80,
                store_particles: true,
                ..ParticleConfig::default()
            },
            20,
            &mut rng,
        )
        .unwrap();
        let kf_last = kf.filtered.last().unwrap()[0];
        for (name, last) in [
            ("pf", pf.filtered.last().unwrap()[0]),
            ("apf", apf.filtered.last().unwrap()[0]),
            ("rpf", rpf.filtered.last().unwrap()[0]),
            ("enkf", enkf.filtered.last().unwrap()[0]),
            ("sis", sis.filtered.last().unwrap()[0]),
            ("upf", upf.filtered.last().unwrap()[0]),
            ("ffbsi", sm.filtered.last().unwrap()[0]),
        ] {
            assert!(
                (last - kf_last).abs() < 1.2,
                "{name} last={last} kf={kf_last}"
            );
        }
        assert!(
            (pf.loglik - kf.loglik).abs() < 5.0,
            "PF loglik {} vs KF {}",
            pf.loglik,
            kf.loglik
        );
        assert_eq!(pf.as_gaussian().filtered.len(), kf.filtered.len());
    }

    #[test]
    fn continuous_discrete_and_sde_ssm() {
        use crate::models::OrnsteinUhlenbeck;
        use crate::sampling::Sampling;
        use crate::simulate::{simulate, SimConfig};

        let sde = OrnsteinUhlenbeck::new(1.2, 0.0, 0.5).unwrap();
        let samp = Sampling::from_terminal(1.5, 80).unwrap();
        let mut rng = seed_rng(8);
        let latent = simulate(&sde, &samp, &[0.0], &mut rng, &SimConfig::default()).unwrap();
        let r = mat_from_row_slice(1, 1, &[0.05]);
        let x0 = col_from_slice(&[0.0]);
        let p0 = mat_from_row_slice(1, 1, &[0.5]);
        let cd = continuous_discrete_ekf(
            &sde,
            |_t, x, out| {
                out[0] = x[0];
            },
            &latent,
            &r,
            &x0,
            &p0,
        )
        .unwrap();
        let wrap = SdeSsm::new(
            sde,
            1,
            |_t, x, out| {
                out[0] = x[0];
            },
            r.clone(),
        )
        .unwrap();
        let ekf = extended_kalman(&wrap, &latent, &x0, &p0).unwrap();
        assert!(cd.loglik.is_finite());
        assert!(ekf.loglik.is_finite());
        assert_eq!(cd.filtered.len(), latent.n_nodes());
        let err = (cd.filtered.last().unwrap()[0] - latent.terminal()[0]).abs();
        assert!(err < 0.8, "cd-ekf err {err}");
    }

    #[test]
    fn nonlinear_ssm_runs() {
        // x⁺ = 0.5 x + 25 x/(1+x²) + 8 cos(1.2 t), y = x²/20
        let q = mat_from_row_slice(1, 1, &[1.0]);
        let r = mat_from_row_slice(1, 1, &[1.0]);
        let model = FnSsm::new(
            1,
            1,
            |t, _dt, x, out| {
                out[0] = 0.5 * x[0] + 25.0 * x[0] / (1.0 + x[0] * x[0]) + 8.0 * (1.2 * t).cos();
            },
            |_t, x, out| {
                out[0] = x[0] * x[0] / 20.0;
            },
            q,
            r,
        )
        .unwrap();
        let n = 16;
        let times: Vec<f64> = (0..=n).map(|i| i as f64).collect();
        let mut vals = vec![0.0; n + 1];
        let mut x = 0.1;
        let mut rng = seed_rng(5);
        for i in 1..=n {
            let t = (i - 1) as f64;
            x = 0.5 * x
                + 25.0 * x / (1.0 + x * x)
                + 8.0 * (1.2 * t).cos()
                + rng.sample(StandardNormal);
            vals[i] = x * x / 20.0 + rng.sample(StandardNormal);
        }
        let obs = Path::new(times, vals, 1).unwrap();
        let x0 = col_from_slice(&[0.0]);
        let p0 = mat_from_row_slice(1, 1, &[1.0]);
        let ekf = extended_kalman(&model, &obs, &x0, &p0).unwrap();
        let ukf = unscented_kalman(&model, &obs, &x0, &p0, UkfParams::default()).unwrap();
        let ckf = cubature_kalman(&model, &obs, &x0, &p0).unwrap();
        let pf = particle_filter(
            &model,
            &obs,
            &x0,
            &p0,
            ParticleConfig {
                n_particles: 200,
                ..ParticleConfig::default()
            },
            &mut rng,
        )
        .unwrap();
        assert!(ekf.loglik.is_finite());
        assert!(ukf.loglik.is_finite());
        assert!(ckf.loglik.is_finite());
        assert!(pf.loglik.is_finite());
        assert_eq!(ekf.filtered.len(), n + 1);
    }
}
