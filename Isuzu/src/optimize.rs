//! Derivative-free box-constrained Nelder–Mead (Pure Rust).

use crate::error::{Error, Result};

/// Options for [`nelder_mead`].
#[derive(Clone, Debug)]
pub struct OptOptions {
    pub max_iter: usize,
    pub ftol: f64,
    pub xtol: f64,
    pub alpha: f64,
    pub gamma: f64,
    pub rho: f64,
    pub sigma: f64,
}

impl Default for OptOptions {
    fn default() -> Self {
        Self {
            max_iter: 400,
            ftol: 1e-10,
            xtol: 1e-8,
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
        }
    }
}

/// Minimizer result.
#[derive(Clone, Debug)]
pub struct OptResult {
    pub x: Vec<f64>,
    pub f: f64,
    pub iters: usize,
    pub converged: bool,
}

fn project(x: &mut [f64], lower: Option<&[f64]>, upper: Option<&[f64]>) {
    for (i, xi) in x.iter_mut().enumerate() {
        if let Some(lo) = lower {
            if *xi < lo[i] {
                *xi = lo[i];
            }
        }
        if let Some(hi) = upper {
            if *xi > hi[i] {
                *xi = hi[i];
            }
        }
    }
}

/// Nelder–Mead simplex search for `min f(x)`.
pub fn nelder_mead(
    f: &dyn Fn(&[f64]) -> f64,
    x0: &[f64],
    lower: Option<&[f64]>,
    upper: Option<&[f64]>,
    opt: OptOptions,
) -> Result<OptResult> {
    let n = x0.len();
    if n == 0 {
        return Err(Error::infer("empty parameter vector"));
    }
    if let Some(lo) = lower {
        if lo.len() != n {
            return Err(Error::infer("lower bound dimension"));
        }
    }
    if let Some(hi) = upper {
        if hi.len() != n {
            return Err(Error::infer("upper bound dimension"));
        }
    }
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    let mut x0c = x0.to_vec();
    project(&mut x0c, lower, upper);
    simplex.push(x0c);
    for i in 0..n {
        let mut y = x0.to_vec();
        let step = if y[i].abs() > 1e-6 {
            0.05 * y[i].abs()
        } else {
            0.05
        };
        y[i] += step;
        project(&mut y, lower, upper);
        if (y[i] - simplex[0][i]).abs() < 1e-14 {
            y[i] -= 2.0 * step;
            project(&mut y, lower, upper);
        }
        simplex.push(y);
    }
    let mut fs: Vec<f64> = simplex.iter().map(|x| f(x)).collect();

    let mut iters = 0;
    let mut converged = false;
    for it in 0..opt.max_iter {
        iters = it + 1;
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&i, &j| {
            fs[i]
                .partial_cmp(&fs[j])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let best = order[0];
        let worst = order[n];
        let second = order[n - 1];

        let frange = (fs[worst] - fs[best]).abs();
        let mut xrange: f64 = 0.0;
        for k in 0..n {
            xrange = xrange.max((simplex[worst][k] - simplex[best][k]).abs());
        }
        if frange < opt.ftol && xrange < opt.xtol {
            converged = true;
            let x = simplex[best].clone();
            return Ok(OptResult {
                x,
                f: fs[best],
                iters,
                converged,
            });
        }

        let mut centroid = vec![0.0; n];
        for &i in order.iter().take(n) {
            for k in 0..n {
                centroid[k] += simplex[i][k];
            }
        }
        for k in 0..n {
            centroid[k] /= n as f64;
        }

        // reflection
        let mut xr = vec![0.0; n];
        for k in 0..n {
            xr[k] = centroid[k] + opt.alpha * (centroid[k] - simplex[worst][k]);
        }
        project(&mut xr, lower, upper);
        let fr = f(&xr);
        if fs[best] <= fr && fr < fs[second] {
            simplex[worst] = xr;
            fs[worst] = fr;
            continue;
        }
        if fr < fs[best] {
            let mut xe = vec![0.0; n];
            for k in 0..n {
                xe[k] = centroid[k] + opt.gamma * (xr[k] - centroid[k]);
            }
            project(&mut xe, lower, upper);
            let fe = f(&xe);
            if fe < fr {
                simplex[worst] = xe;
                fs[worst] = fe;
            } else {
                simplex[worst] = xr;
                fs[worst] = fr;
            }
            continue;
        }
        // contraction
        let mut xc = vec![0.0; n];
        for k in 0..n {
            xc[k] = centroid[k] + opt.rho * (simplex[worst][k] - centroid[k]);
        }
        project(&mut xc, lower, upper);
        let fc = f(&xc);
        if fc < fs[worst] {
            simplex[worst] = xc;
            fs[worst] = fc;
            continue;
        }
        // shrink toward best
        for i in 0..=n {
            if i == best {
                continue;
            }
            for k in 0..n {
                simplex[i][k] = simplex[best][k] + opt.sigma * (simplex[i][k] - simplex[best][k]);
            }
            project(&mut simplex[i], lower, upper);
            fs[i] = f(&simplex[i]);
        }
    }
    let mut order: Vec<usize> = (0..=n).collect();
    order.sort_by(|&i, &j| {
        fs[i]
            .partial_cmp(&fs[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let best = order[0];
    Ok(OptResult {
        x: simplex[best].clone(),
        f: fs[best],
        iters,
        converged,
    })
}

/// One-dimensional golden-section search on `[lo, hi]`.
pub fn golden_section(
    f: &dyn Fn(f64) -> f64,
    mut lo: f64,
    mut hi: f64,
    tol: f64,
) -> Result<(f64, f64)> {
    if hi <= lo {
        return Err(Error::infer("golden section requires hi > lo"));
    }
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let mut c = hi - (hi - lo) / phi;
    let mut d = lo + (hi - lo) / phi;
    let mut fc = f(c);
    let mut fd = f(d);
    for _ in 0..200 {
        if (hi - lo).abs() < tol {
            break;
        }
        if fc < fd {
            hi = d;
            d = c;
            fd = fc;
            c = hi - (hi - lo) / phi;
            fc = f(c);
        } else {
            lo = c;
            c = d;
            fc = fd;
            d = lo + (hi - lo) / phi;
            fd = f(d);
        }
    }
    let x = 0.5 * (lo + hi);
    Ok((x, f(x)))
}

/// Options for [`lbfgs_b`].
#[derive(Clone, Debug)]
pub struct LbfgsOptions {
    pub max_iter: usize,
    pub m: usize,
    pub ftol: f64,
    pub gtol: f64,
    pub fd_step: f64,
}

impl Default for LbfgsOptions {
    fn default() -> Self {
        Self {
            max_iter: 200,
            m: 8,
            ftol: 1e-10,
            gtol: 1e-6,
            fd_step: 1e-6,
        }
    }
}

fn project_box(x: &mut [f64], lower: Option<&[f64]>, upper: Option<&[f64]>) {
    project(x, lower, upper);
}

fn finite_diff_grad(f: &dyn Fn(&[f64]) -> f64, x: &[f64], step: f64, g: &mut [f64], f0: f64) {
    let mut x2 = x.to_vec();
    for i in 0..x.len() {
        let h = step * (1.0 + x[i].abs());
        x2[i] = x[i] + h;
        let fp = f(&x2);
        x2[i] = x[i];
        g[i] = (fp - f0) / h;
    }
}

fn projected_grad_norm(x: &[f64], g: &[f64], lower: Option<&[f64]>, upper: Option<&[f64]>) -> f64 {
    let mut s = 0.0;
    for i in 0..x.len() {
        let mut gi = g[i];
        if let Some(lo) = lower {
            if x[i] <= lo[i] && gi > 0.0 {
                gi = 0.0;
            }
        }
        if let Some(hi) = upper {
            if x[i] >= hi[i] && gi < 0.0 {
                gi = 0.0;
            }
        }
        s += gi * gi;
    }
    s.sqrt()
}

/// Limited-memory BFGS with box projection (Byrd–Lu–Nocedal–Zhu style).
///
/// `grad` is optional; when omitted the gradient is a forward finite difference.
/// The box is enforced by projection at every trial point. This is the
/// estimator used by QMLE / SSM MLE when [`OptOptions`] is not enough
/// (Nelder–Mead degenerates on a bound face).
pub fn lbfgs_b(
    f: &dyn Fn(&[f64]) -> f64,
    grad: Option<&dyn Fn(&[f64], &mut [f64])>,
    x0: &[f64],
    lower: Option<&[f64]>,
    upper: Option<&[f64]>,
    opt: LbfgsOptions,
) -> Result<OptResult> {
    let n = x0.len();
    if n == 0 {
        return Err(Error::infer("empty parameter vector"));
    }
    let mut x = x0.to_vec();
    project_box(&mut x, lower, upper);
    let mut g = vec![0.0; n];
    let mut fcur = f(&x);
    if let Some(gr) = grad {
        gr(&x, &mut g);
    } else {
        finite_diff_grad(f, &x, opt.fd_step, &mut g, fcur);
    }
    let mut s_hist: Vec<Vec<f64>> = Vec::new();
    let mut y_hist: Vec<Vec<f64>> = Vec::new();
    let mut rho_hist: Vec<f64> = Vec::new();
    let mut iters = 0;
    let mut converged = false;
    for it in 0..opt.max_iter {
        iters = it + 1;
        if projected_grad_norm(&x, &g, lower, upper) < opt.gtol {
            converged = true;
            break;
        }
        // Two-loop recursion for H g.
        let mut q = g.clone();
        let mut alpha = vec![0.0; s_hist.len()];
        for i in (0..s_hist.len()).rev() {
            let mut dot = 0.0;
            for k in 0..n {
                dot += s_hist[i][k] * q[k];
            }
            alpha[i] = rho_hist[i] * dot;
            for k in 0..n {
                q[k] -= alpha[i] * y_hist[i][k];
            }
        }
        let mut gamma = 1.0;
        if let (Some(s), Some(y)) = (s_hist.last(), y_hist.last()) {
            let mut ys = 0.0;
            let mut yy = 0.0;
            for k in 0..n {
                ys += y[k] * s[k];
                yy += y[k] * y[k];
            }
            if yy > 0.0 {
                gamma = ys / yy;
            }
        }
        let mut r = vec![0.0; n];
        for k in 0..n {
            r[k] = gamma * q[k];
        }
        for i in 0..s_hist.len() {
            let mut dot = 0.0;
            for k in 0..n {
                dot += y_hist[i][k] * r[k];
            }
            let beta = rho_hist[i] * dot;
            for k in 0..n {
                r[k] += s_hist[i][k] * (alpha[i] - beta);
            }
        }
        // Search direction: projected steepest with L-BFGS metric.
        let mut d = vec![0.0; n];
        for k in 0..n {
            d[k] = -r[k];
        }
        let mut gdotd = 0.0;
        for k in 0..n {
            gdotd += g[k] * d[k];
        }
        if !gdotd.is_finite() || gdotd >= 0.0 {
            for k in 0..n {
                d[k] = -g[k];
            }
        }
        // Armijo line search on the projected path.
        let mut step = 1.0;
        let mut accepted = false;
        let mut xnew = x.clone();
        let mut fnew = fcur;
        for _ in 0..20 {
            for k in 0..n {
                xnew[k] = x[k] + step * d[k];
            }
            project_box(&mut xnew, lower, upper);
            fnew = f(&xnew);
            let mut dg = 0.0;
            for k in 0..n {
                dg += g[k] * (xnew[k] - x[k]);
            }
            if fnew <= fcur + 1e-4 * dg && fnew.is_finite() {
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            // Gradient step.
            step = 1e-3;
            for k in 0..n {
                xnew[k] = x[k] - step * g[k];
            }
            project_box(&mut xnew, lower, upper);
            fnew = f(&xnew);
        }
        let mut gnew = vec![0.0; n];
        if let Some(gr) = grad {
            gr(&xnew, &mut gnew);
        } else {
            finite_diff_grad(f, &xnew, opt.fd_step, &mut gnew, fnew);
        }
        let mut s = vec![0.0; n];
        let mut y = vec![0.0; n];
        let mut ys = 0.0;
        for k in 0..n {
            s[k] = xnew[k] - x[k];
            y[k] = gnew[k] - g[k];
            ys += y[k] * s[k];
        }
        if ys > 1e-16 {
            if s_hist.len() == opt.m {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            s_hist.push(s);
            y_hist.push(y);
            rho_hist.push(1.0 / ys);
        }
        if (fcur - fnew).abs() < opt.ftol * (1.0 + fcur.abs())
            && projected_grad_norm(&xnew, &gnew, lower, upper) < opt.gtol
        {
            x = xnew;
            fcur = fnew;
            converged = true;
            break;
        }
        x = xnew;
        fcur = fnew;
        g = gnew;
    }
    Ok(OptResult {
        x,
        f: fcur,
        iters,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nelder_rosenbrock() {
        let f = |p: &[f64]| {
            let (x, y) = (p[0], p[1]);
            (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2)
        };
        let r = nelder_mead(&f, &[0.0, 0.0], None, None, OptOptions::default()).unwrap();
        assert!(r.f < 1e-6);
        assert!((r.x[0] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn lbfgs_b_rosenbrock_and_box() {
        let f = |p: &[f64]| {
            let (x, y) = (p[0], p[1]);
            (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2)
        };
        let r = lbfgs_b(&f, None, &[0.0, 0.0], None, None, LbfgsOptions::default()).unwrap();
        assert!(r.f < 1e-5, "f={}", r.f);
        let boxed = lbfgs_b(
            &f,
            None,
            &[0.2, 0.2],
            Some(&[0.0, 0.0]),
            Some(&[0.5, 0.5]),
            LbfgsOptions::default(),
        )
        .unwrap();
        assert!(boxed.x[0] <= 0.5 + 1e-12);
        assert!(boxed.x[1] <= 0.5 + 1e-12);
        assert!(boxed.f < f(&[0.2, 0.2]));
    }
}
