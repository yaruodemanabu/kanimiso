use mayoi_no_mori::{DenseMatrix, ForestOptions, RandomForestClassifier};
use oldwood::ClassificationCriterion;

fn main() -> Result<(), mayoi_no_mori::Error> {
    let x = DenseMatrix::from_row_major(
        6,
        2,
        vec![0.0, 0.1, 0.1, 0.0, 0.2, 0.2, 0.8, 0.9, 0.9, 0.8, 1.0, 1.0],
    )?;
    let target = [10, 10, 10, 42, 42, 42];
    let fitted = RandomForestClassifier::new(
        ForestOptions {
            trees: 64,
            seed: 7,
            ..ForestOptions::default()
        },
        ClassificationCriterion::Gini,
    )
    .fit(&x, &target, None)?;
    println!("{:?}", fitted.predict(&x)?);
    Ok(())
}
