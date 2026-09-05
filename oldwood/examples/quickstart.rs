use oldwood::{
    ClassificationCriterion, DecisionTreeClassifier, DenseMatrix, MatrixView, TreeOptions,
};

fn main() -> oldwood::Result<()> {
    let x = DenseMatrix::from_row_major(6, 1, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])?;
    let y = [10, 10, 10, 20, 20, 20];
    let weights = [1.0, 2.0, 1.0, 1.0, 2.0, 1.0];
    let tree = DecisionTreeClassifier::new(
        ClassificationCriterion::Gini,
        TreeOptions {
            max_depth: Some(3),
            ..TreeOptions::default()
        },
    )
    .fit(&x, &y, Some(&weights))?;

    println!("classes: {:?}", tree.classes());
    println!("predictions: {:?}", tree.predict(&x)?);
    println!("probability rows: {}", tree.predict_proba(&x)?.nrows());
    println!("arena nodes: {}", tree.nodes().len());
    Ok(())
}
