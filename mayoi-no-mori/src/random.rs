//! Small sampling helpers over Isuzu's independently tested `ChaCha8` stream.

use amatsuki::Rng;

pub(crate) fn below<R: Rng + ?Sized>(rng: &mut R, upper: usize) -> usize {
    debug_assert!(upper > 0);
    let upper = u64::try_from(upper).expect("supported pointer widths fit into u64");
    let zone = u64::MAX - u64::MAX % upper;
    loop {
        let value = rng.next_u64();
        if value < zone {
            return usize::try_from(value % upper)
                .expect("the remainder is below the original usize bound");
        }
    }
}

pub(crate) fn shuffle<R: Rng + ?Sized, T>(rng: &mut R, values: &mut [T]) {
    for index in (1..values.len()).rev() {
        values.swap(index, below(rng, index + 1));
    }
}

pub(crate) fn sample_rows<R: Rng + ?Sized>(
    rng: &mut R,
    rows: usize,
    count: usize,
    replacement: bool,
) -> Vec<usize> {
    if replacement {
        return (0..count).map(|_| below(rng, rows)).collect();
    }
    let mut indices: Vec<usize> = (0..rows).collect();
    for index in 0..count {
        let selected = index + below(rng, rows - index);
        indices.swap(index, selected);
    }
    indices.truncate(count);
    indices
}

pub(crate) fn soft_threshold(value: f64, l1: f64) -> f64 {
    if value > l1 {
        value - l1
    } else if value < -l1 {
        value + l1
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use amatsuki::seed_rng;

    #[test]
    fn bounded_sampling_has_a_fixed_seed_replay() {
        let mut rng = seed_rng(2_026);
        let draws: Vec<usize> = (0..8).map(|_| below(&mut rng, 7)).collect();
        assert_eq!(draws, [5, 3, 5, 3, 6, 4, 5, 5]);
        assert!(draws.iter().all(|&draw| draw < 7));

        let mut values: Vec<usize> = (0..8).collect();
        shuffle(&mut seed_rng(2_026), &mut values);
        assert_eq!(values, [5, 6, 1, 7, 4, 2, 3, 0]);
        let mut sorted = values.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn row_sampling_respects_count_replacement_and_endpoints() {
        assert!(sample_rows(&mut seed_rng(7), 5, 0, false).is_empty());

        let all = sample_rows(&mut seed_rng(7), 5, 5, false);
        assert_eq!(all.len(), 5);
        let mut sorted = all.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);

        let subset = sample_rows(&mut seed_rng(9), 17, 11, false);
        assert_eq!(subset.len(), 11);
        let mut unique = subset.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), subset.len());
        assert!(subset.iter().all(|&row| row < 17));

        let replacement = sample_rows(&mut seed_rng(11), 3, 64, true);
        assert_eq!(replacement.len(), 64);
        assert!(replacement.iter().all(|&row| row < 3));
    }
}
