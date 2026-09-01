//! Log-space Viterbi path.

/// Most likely state path and its joint log-probability.
pub(crate) fn viterbi_path(
    log_start: &[f64],
    log_trans: &[Vec<f64>],
    log_emit: &[Vec<f64>],
) -> (Vec<usize>, f64) {
    let t_len = log_emit.len();
    let s = log_start.len();
    if t_len == 0 || s == 0 {
        return (Vec::new(), f64::NEG_INFINITY);
    }
    let mut delta = vec![vec![f64::NEG_INFINITY; s]; t_len];
    let mut psi = vec![vec![0usize; s]; t_len];
    for j in 0..s {
        delta[0][j] = log_start[j] + log_emit[0][j];
    }
    for t in 1..t_len {
        for j in 0..s {
            let mut best = f64::NEG_INFINITY;
            let mut arg = 0usize;
            for i in 0..s {
                let v = delta[t - 1][i] + log_trans[i][j];
                if v > best {
                    best = v;
                    arg = i;
                }
            }
            delta[t][j] = best + log_emit[t][j];
            psi[t][j] = arg;
        }
    }
    let mut last = 0usize;
    let mut best = f64::NEG_INFINITY;
    for j in 0..s {
        if delta[t_len - 1][j] > best {
            best = delta[t_len - 1][j];
            last = j;
        }
    }
    let mut path = vec![0usize; t_len];
    path[t_len - 1] = last;
    for t in (1..t_len).rev() {
        path[t - 1] = psi[t][path[t]];
    }
    (path, best)
}
