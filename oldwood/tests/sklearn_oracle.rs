#![allow(clippy::float_cmp)]

use oldwood::{
    ClassificationCriterion, DecisionTreeClassifier, DecisionTreeRegressor, DenseMatrix,
    RegressionCriterion, TreeOptions,
};

const FIXTURE: &str = include_str!("../golden/sklearn_cart.csv");
const EXPECTED_HEADER: &str = "case,task,criterion,max_depth,random_state,role,row,x0,x1,target,weight,prediction,class0,class1,prob0,prob1";
// The initial CPython/scikit-learn 1.7.2 replay on 2026-09-04 measured zero
// ULP for every probability and regression prediction. The committed binary64
// fixture therefore needs no nonzero tolerance.
const PROBABILITY_ABSOLUTE_TOLERANCE: f64 = 0.0;
const REGRESSION_ABSOLUTE_TOLERANCE: f64 = 0.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Task {
    Classifier,
    Regressor,
}

#[derive(Debug)]
enum Target {
    Class(usize),
    Regression(f64),
}

#[derive(Clone, Copy, Debug)]
enum Prediction {
    Class(usize),
    Regression(f64),
}

#[derive(Debug)]
struct Probe {
    features: [f64; 2],
    prediction: Prediction,
    classes: Option<[usize; 2]>,
    probabilities: Option<[f64; 2]>,
}

#[derive(Debug)]
struct Case {
    name: String,
    task: Task,
    criterion: String,
    max_depth: usize,
    random_state: u64,
    train_x: Vec<[f64; 2]>,
    targets: Vec<Target>,
    weights: Vec<f64>,
    probes: Vec<Probe>,
}

impl Case {
    fn from_fields(fields: &[&str]) -> Self {
        Self {
            name: fields[0].to_owned(),
            task: parse_task(fields[1]),
            criterion: fields[2].to_owned(),
            max_depth: parse(fields[3]),
            random_state: parse(fields[4]),
            train_x: Vec::new(),
            targets: Vec::new(),
            weights: Vec::new(),
            probes: Vec::new(),
        }
    }

    fn verify_metadata(&self, fields: &[&str]) {
        assert_eq!(self.task, parse_task(fields[1]), "{} task", self.name);
        assert_eq!(self.criterion, fields[2], "{} criterion", self.name);
        assert_eq!(self.max_depth, parse(fields[3]), "{} depth", self.name);
        assert_eq!(
            self.random_state,
            parse(fields[4]),
            "{} random state",
            self.name
        );
    }

    fn push(&mut self, fields: &[&str]) {
        self.verify_metadata(fields);
        let features = [parse(fields[7]), parse(fields[8])];
        match fields[5] {
            "train" => {
                assert_eq!(parse::<usize>(fields[6]), self.train_x.len());
                self.train_x.push(features);
                self.weights.push(parse(fields[10]));
                self.targets.push(match self.task {
                    Task::Classifier => Target::Class(parse(fields[9])),
                    Task::Regressor => Target::Regression(parse(fields[9])),
                });
                assert!(fields[11..].iter().all(|field| field.is_empty()));
            }
            "probe" => {
                assert_eq!(parse::<usize>(fields[6]), self.probes.len());
                assert!(fields[9].is_empty() && fields[10].is_empty());
                let (prediction, classes, probabilities) = match self.task {
                    Task::Classifier => (
                        Prediction::Class(parse(fields[11])),
                        Some([parse(fields[12]), parse(fields[13])]),
                        Some([parse(fields[14]), parse(fields[15])]),
                    ),
                    Task::Regressor => {
                        assert!(fields[12..].iter().all(|field| field.is_empty()));
                        (Prediction::Regression(parse(fields[11])), None, None)
                    }
                };
                self.probes.push(Probe {
                    features,
                    prediction,
                    classes,
                    probabilities,
                });
            }
            role => panic!("unknown fixture role {role}"),
        }
    }
}

#[test]
fn scikit_learn_weighted_cart_predictions_match() {
    assert!(FIXTURE.contains("# schema=1\n"));
    assert!(FIXTURE.contains("# scikit_learn=1.7.2\n"));
    let cases = parse_fixture();
    assert_eq!(cases.len(), 3);

    let mut maximum_probability_error = 0.0_f64;
    let mut maximum_regression_error = 0.0_f64;
    for case in cases {
        assert_eq!(case.random_state, 1729);
        let options = TreeOptions {
            max_depth: Some(case.max_depth),
            ..TreeOptions::default()
        };
        let train = matrix(&case.train_x);
        let probe_features: Vec<[f64; 2]> =
            case.probes.iter().map(|probe| probe.features).collect();
        let probes = matrix(&probe_features);
        match case.task {
            Task::Classifier => {
                let targets: Vec<usize> = case
                    .targets
                    .iter()
                    .map(|target| match target {
                        Target::Class(value) => *value,
                        Target::Regression(_) => panic!("mixed target kinds"),
                    })
                    .collect();
                let criterion = match case.criterion.as_str() {
                    "gini" => ClassificationCriterion::Gini,
                    "entropy" => ClassificationCriterion::Entropy,
                    value => panic!("unknown classification criterion {value}"),
                };
                let fitted = DecisionTreeClassifier::new(criterion, options)
                    .fit(&train, &targets, Some(&case.weights))
                    .unwrap();
                let predictions = fitted.predict(&probes).unwrap();
                let probabilities = fitted.predict_proba(&probes).unwrap();
                for (row, expected) in case.probes.iter().enumerate() {
                    let Prediction::Class(expected_prediction) = expected.prediction else {
                        panic!("mixed prediction kinds");
                    };
                    assert_eq!(predictions[row], expected_prediction, "{}", case.name);
                    assert_eq!(fitted.classes(), expected.classes.unwrap(), "{}", case.name);
                    for class in 0..2 {
                        let error = (probabilities.as_slice()[row * 2 + class]
                            - expected.probabilities.unwrap()[class])
                            .abs();
                        maximum_probability_error = maximum_probability_error.max(error);
                    }
                }
            }
            Task::Regressor => {
                assert_eq!(case.criterion, "squared_error");
                let targets: Vec<f64> = case
                    .targets
                    .iter()
                    .map(|target| match target {
                        Target::Regression(value) => *value,
                        Target::Class(_) => panic!("mixed target kinds"),
                    })
                    .collect();
                let fitted = DecisionTreeRegressor::new(RegressionCriterion::SquaredError, options)
                    .fit(&train, &targets, Some(&case.weights))
                    .unwrap();
                for (actual, expected) in fitted.predict(&probes).unwrap().iter().zip(&case.probes)
                {
                    let Prediction::Regression(expected_prediction) = expected.prediction else {
                        panic!("mixed prediction kinds");
                    };
                    maximum_regression_error =
                        maximum_regression_error.max((actual - expected_prediction).abs());
                }
            }
        }
    }

    assert!(maximum_probability_error <= PROBABILITY_ABSOLUTE_TOLERANCE);
    assert!(maximum_regression_error <= REGRESSION_ABSOLUTE_TOLERANCE);
}

fn parse_fixture() -> Vec<Case> {
    let mut cases: Vec<Case> = Vec::new();
    let mut saw_header = false;
    for line in FIXTURE.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        if !saw_header {
            assert_eq!(line, EXPECTED_HEADER);
            saw_header = true;
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        assert_eq!(fields.len(), 16, "invalid fixture row: {line}");
        if cases.last().is_none_or(|case| case.name != fields[0]) {
            cases.push(Case::from_fields(&fields));
        }
        cases.last_mut().unwrap().push(&fields);
    }
    assert!(saw_header);
    for case in &cases {
        assert!(
            !case.train_x.is_empty(),
            "{} has no training rows",
            case.name
        );
        assert!(!case.probes.is_empty(), "{} has no probe rows", case.name);
        assert_eq!(case.train_x.len(), case.targets.len());
        assert_eq!(case.train_x.len(), case.weights.len());
    }
    cases
}

fn matrix(rows: &[[f64; 2]]) -> DenseMatrix {
    DenseMatrix::from_row_major(
        rows.len(),
        2,
        rows.iter().flat_map(|row| row.iter().copied()).collect(),
    )
    .unwrap()
}

fn parse<T: core::str::FromStr>(text: &str) -> T
where
    T::Err: core::fmt::Debug,
{
    text.parse().unwrap()
}

fn parse_task(text: &str) -> Task {
    match text {
        "classifier" => Task::Classifier,
        "regressor" => Task::Regressor,
        value => panic!("unknown task {value}"),
    }
}
