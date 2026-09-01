//! Bag-of-words and TF–IDF (sklearn `feature_extraction.text`).
//!
//! Tokenization is whitespace / punctuation splitting. A vocabulary that
//! collapses to a single token records [`IssueCode::NearZeroVariance`]. An
//! empty corpus is [`IssueCode::EmptyMatrix`].

use crate::context::FitCtx;
use crate::data::{Matrix, Vector};
use crate::traits::{FitUnsupervised, Transform};
use crate::validate::inspect_xy;
use ojizou_san::Session;
use signlred::{Issue, IssueCode, Meaninglessness, Qualified, Result};
use std::collections::BTreeMap;

fn tokenize(doc: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in doc.chars() {
        if c.is_alphanumeric() {
            cur.extend(c.to_lowercase());
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Count-vectorizer: documents → term-count matrix.
#[derive(Clone, Debug, Default)]
pub(crate) struct CountVectorizer {
    /// Ignore tokens that appear in fewer than this many documents.
    pub min_df: usize,
    vocab: Vec<String>,
    fitted: bool,
}

impl CountVectorizer {
    /// Default tokenizer (`min_df = 1`).
    pub(crate) fn new() -> Self {
        Self {
            min_df: 1,
            vocab: Vec::new(),
            fitted: false,
        }
    }

    /// Sorted vocabulary after `fit_docs`.
    pub(crate) fn vocabulary(&self) -> &[String] {
        &self.vocab
    }

    /// Learn the vocabulary from `docs`.
    pub(crate) fn fit_docs(&mut self, docs: &[&str], session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        if docs.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::EmptyMatrix)
                    .message("CountVectorizer received 0 documents")
                    .build(),
            );
            self.vocab.clear();
            self.fitted = true;
            return ctx.finish(self.clone());
        }
        let mut df: BTreeMap<String, usize> = BTreeMap::new();
        for d in docs {
            let mut seen = BTreeMap::new();
            for tok in tokenize(d) {
                seen.insert(tok, ());
            }
            for t in seen.into_keys() {
                *df.entry(t).or_insert(0) += 1;
            }
        }
        let min_df = self.min_df.max(1);
        self.vocab = df
            .into_iter()
            .filter(|(_, c)| *c >= min_df)
            .map(|(t, _)| t)
            .collect();
        if self.vocab.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::NearZeroVariance)
                    .message("vocabulary is empty after min_df filtering")
                    .meaninglessness(Meaninglessness::vacuous(
                        "count matrix",
                        "no token survived the document-frequency floor",
                        "lower min_df or provide documents with shared tokens",
                    ))
                    .build(),
            );
        }
        if self.vocab.len() == 1 {
            ctx.push(
                Issue::builder(IssueCode::NearZeroVariance)
                    .message("vocabulary collapsed to a single token")
                    .build(),
            );
        }
        self.fitted = true;
        ctx.finish(self.clone())
    }

    /// Transform `docs` with the fitted vocabulary.
    pub(crate) fn transform_docs(
        &self,
        docs: &[&str],
        session: &Session,
    ) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(Matrix::zeros(docs.len(), 0));
        }
        let p = self.vocab.len();
        let index: BTreeMap<&str, usize> = self
            .vocab
            .iter()
            .enumerate()
            .map(|(i, t)| (t.as_str(), i))
            .collect();
        let x = Matrix::from_fn(docs.len(), p, |i, j| {
            let mut c = 0.0;
            for tok in tokenize(docs[i]) {
                if index.get(tok.as_str()) == Some(&j) {
                    c += 1.0;
                }
            }
            c
        });
        ctx.finish(x)
    }
}

/// TF–IDF transformer on a nonnegative count matrix (sklearn `TfidfTransformer`).
#[derive(Clone, Debug)]
pub(crate) struct TfidfTransformer {
    /// Smooth IDF by adding 1 to document and document-frequency counts.
    pub smooth_idf: bool,
    idf: Vector,
    fitted: bool,
}

impl Default for TfidfTransformer {
    fn default() -> Self {
        Self {
            smooth_idf: true,
            idf: Vector::zeros(0),
            fitted: false,
        }
    }
}

impl TfidfTransformer {
    /// Default smoothed IDF.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Fitted IDF vector.
    pub(crate) fn idf(&self) -> &Vector {
        &self.idf
    }
}

impl FitUnsupervised for TfidfTransformer {
    type Fitted = Self;
    fn fit_unsupervised(&self, x: &Matrix, session: &Session) -> Result<Qualified<Self>> {
        let mut this = self.clone();
        let mut ctx = FitCtx::with_session(session.clone());
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        let (n, p) = x.shape();
        let mut clipped = 0usize;
        let mut df = vec![0.0f64; p];
        for j in 0..p {
            for i in 0..n {
                let v = x.get(i, j);
                if v < 0.0 {
                    clipped += 1;
                }
                if v > 0.0 {
                    df[j] += 1.0;
                }
            }
        }
        if clipped > 0 {
            ctx.push(
                Issue::builder(IssueCode::InconsistentSystem)
                    .message(format!(
                        "TfidfTransformer ignored {clipped} negative counts"
                    ))
                    .build(),
            );
        }
        let nf = n as f64;
        this.idf = Vector::from_iter((0..p).map(|j| {
            let (num, den) = if this.smooth_idf {
                (nf + 1.0, df[j] + 1.0)
            } else {
                (nf.max(1.0), df[j].max(1.0))
            };
            (num / den).ln() + 1.0
        }));
        if p > 0 && df.iter().all(|c| *c <= 0.0) {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("every term has document frequency 0")
                    .meaninglessness(Meaninglessness::vacuous(
                        "TF–IDF",
                        "the count matrix is identically zero",
                        "fit on a corpus that contains tokens",
                    ))
                    .build(),
            );
        }
        this.fitted = true;
        ctx.finish(this.clone())
    }
}

impl Transform for TfidfTransformer {
    fn transform(&self, x: &Matrix, session: &Session) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        inspect_xy(&mut ctx.report, x, None, &ctx.policy);
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::PartialFitBeforeInit).build());
            return ctx.finish(x.clone());
        }
        if x.ncols() != self.idf.len() {
            ctx.push(
                Issue::builder(IssueCode::DimensionMismatch)
                    .message("TfidfTransformer column count ≠ fitted vocabulary")
                    .build(),
            );
        }
        let p = x.ncols().min(self.idf.len());
        let mut out = Matrix::from_fn(x.nrows(), x.ncols(), |i, j| {
            if j < p {
                x.get(i, j).max(0.0) * self.idf[j]
            } else {
                0.0
            }
        });
        for i in 0..out.nrows() {
            let mut nrm = 0.0;
            for j in 0..out.ncols() {
                let v = out.get(i, j);
                nrm += v * v;
            }
            let nrm = nrm.sqrt();
            if nrm > ctx.policy.near_zero_variance {
                for j in 0..out.ncols() {
                    out.set(i, j, out.get(i, j) / nrm);
                }
            }
        }
        ctx.finish(out)
    }
}

/// Count + TF–IDF in one step (sklearn `TfidfVectorizer`).
#[derive(Clone, Debug, Default)]
pub(crate) struct TfidfVectorizer {
    /// Document-frequency floor forwarded to [`CountVectorizer`].
    pub min_df: usize,
    counts: CountVectorizer,
    tfidf: TfidfTransformer,
}

impl TfidfVectorizer {
    /// Default vectorizer.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Vocabulary after `fit_docs`.
    pub(crate) fn vocabulary(&self) -> &[String] {
        self.counts.vocabulary()
    }

    /// Fit vocabulary and IDF on `docs`.
    pub(crate) fn fit_docs(&mut self, docs: &[&str], session: &Session) -> Result<Qualified<Self>> {
        self.counts.min_df = self.min_df.max(1);
        let q = self.counts.fit_docs(docs, &session.child("count"))?;
        self.counts = q.value;
        let x = self
            .counts
            .transform_docs(docs, &session.child("count-x"))?
            .value;
        let t = self.tfidf.fit_unsupervised(&x, &session.child("tfidf"))?;
        self.tfidf = t.value;
        let mut ctx = FitCtx::with_session(session.clone());
        for issue in q.report.issues() {
            ctx.push(issue.clone());
        }
        for issue in t.report.issues() {
            ctx.push(issue.clone());
        }
        ctx.finish(self.clone())
    }

    /// Transform `docs` to row-normalized TF–IDF.
    pub(crate) fn transform_docs(
        &self,
        docs: &[&str],
        session: &Session,
    ) -> Result<Qualified<Matrix>> {
        let x = self
            .counts
            .transform_docs(docs, &session.child("count-t"))?
            .value;
        self.tfidf.transform(&x, &session.child("tfidf-t"))
    }
}

fn fnv1a(tok: &str) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for b in tok.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Signed hashing trick (sklearn `HashingVectorizer`). No vocabulary is stored.
#[derive(Clone, Debug)]
pub(crate) struct HashingVectorizer {
    /// Number of hash bins.
    pub n_features: usize,
}

impl Default for HashingVectorizer {
    fn default() -> Self {
        Self { n_features: 16 }
    }
}

impl HashingVectorizer {
    /// Hasher with `n_features` bins.
    pub(crate) fn new(n_features: usize) -> Self {
        Self {
            n_features: n_features.max(1),
        }
    }

    /// Transform `docs` (stateless).
    pub(crate) fn transform_docs(
        &self,
        docs: &[&str],
        session: &Session,
    ) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.clone());
        if docs.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::EmptyMatrix)
                    .message("HashingVectorizer received 0 documents")
                    .build(),
            );
            return ctx.finish(Matrix::zeros(0, self.n_features.max(1)));
        }
        let p = self.n_features.max(1);
        if p < 8 && docs.len() > 4 {
            ctx.push(
                Issue::builder(IssueCode::NearZeroVariance)
                    .severity(signlred::Severity::Advisory)
                    .message(format!(
                        "HashingVectorizer n_features={p} is small relative to the corpus; collisions are likely"
                    ))
                    .build(),
            );
        }
        let x = Matrix::from_fn(docs.len(), p, |i, j| {
            let mut s = 0.0;
            for tok in tokenize(docs[i]) {
                let h = fnv1a(&tok);
                let bin = (h >> 1) as usize % p;
                if bin == j {
                    s += if h & 1 == 1 { 1.0 } else { -1.0 };
                }
            }
            s
        });
        ctx.finish(x)
    }
}

/// Dict-of-features vectorizer (sklearn `DictVectorizer`).
///
/// Each document is a whitespace-separated `key=value` record. Keys become
/// columns; missing keys are 0. An empty key alphabet is vacuous.
#[derive(Clone, Debug, Default)]
pub(crate) struct DictVectorizer {
    vocab: Vec<String>,
    fitted: bool,
}

impl DictVectorizer {
    /// Empty vectorizer.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Sorted feature keys.
    pub(crate) fn vocabulary(&self) -> &[String] {
        &self.vocab
    }

    fn parse_record(doc: &str) -> BTreeMap<String, f64> {
        let mut out = BTreeMap::new();
        for tok in doc.split_whitespace() {
            if let Some((k, v)) = tok.split_once('=') {
                if k.is_empty() {
                    continue;
                }
                let val = v.parse::<f64>().unwrap_or(1.0);
                *out.entry(k.to_string()).or_insert(0.0) += val;
            } else if !tok.is_empty() {
                *out.entry(tok.to_string()).or_insert(0.0) += 1.0;
            }
        }
        out
    }

    /// Learn keys from `key=value` records.
    pub(crate) fn fit_docs(&mut self, docs: &[&str], session: &Session) -> Result<Qualified<Self>> {
        let mut ctx = FitCtx::with_session(session.clone());
        if docs.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::EmptyMatrix)
                    .message("DictVectorizer received 0 records")
                    .build(),
            );
            self.vocab.clear();
            self.fitted = true;
            return ctx.finish(self.clone());
        }
        let mut keys = BTreeMap::<String, ()>::new();
        for d in docs {
            for k in Self::parse_record(d).into_keys() {
                keys.insert(k, ());
            }
        }
        self.vocab = keys.into_keys().collect();
        if self.vocab.is_empty() {
            ctx.push(
                Issue::builder(IssueCode::MeaninglessFit)
                    .message("DictVectorizer saw no keys")
                    .meaninglessness(Meaninglessness::vacuous(
                        "dict features",
                        "an empty key alphabet cannot be vectorized",
                        "supply key=value records",
                    ))
                    .build(),
            );
        }
        self.fitted = true;
        ctx.finish(self.clone())
    }

    /// Map records to a dense `n × |vocab|` matrix.
    pub(crate) fn transform_docs(
        &self,
        docs: &[&str],
        session: &Session,
    ) -> Result<Qualified<Matrix>> {
        let mut ctx = FitCtx::with_session(session.child("transform"));
        if !self.fitted {
            ctx.push(Issue::builder(IssueCode::StaleState).build());
            return ctx.finish(Matrix::zeros(docs.len(), 0));
        }
        let p = self.vocab.len();
        let out = Matrix::from_fn(docs.len(), p, |i, j| {
            let rec = Self::parse_record(docs[i]);
            rec.get(&self.vocab[j]).copied().unwrap_or(0.0)
        });
        ctx.finish(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojizou_san::Session;

    #[test]
    fn counts_and_tfidf_on_two_docs() {
        let docs = ["red fox red", "blue fox"];
        let mut cv = CountVectorizer::new();
        cv.fit_docs(&docs, &Session::new("cv", "fit")).unwrap();
        assert!(cv.vocabulary().contains(&"fox".to_string()));
        let x = cv
            .transform_docs(&docs, &Session::new("cv", "t"))
            .unwrap()
            .value;
        assert_eq!(x.nrows(), 2);
        assert!(x.ncols() >= 3);
        let mut tf = TfidfTransformer::new();
        tf.fit_unsupervised(&x, &Session::new("tf", "fit")).unwrap();
        let z = tf.transform(&x, &Session::new("tf", "t")).unwrap().value;
        assert_eq!(z.shape(), x.shape());
        let n0: f64 = (0..z.ncols()).map(|j| z.get(0, j) * z.get(0, j)).sum();
        assert!((n0.sqrt() - 1.0).abs() < 1e-10, "n0={n0}");
        let mut tv = TfidfVectorizer::new();
        tv.fit_docs(&docs, &Session::new("tv", "fit")).unwrap();
        let z2 = tv
            .transform_docs(&docs, &Session::new("tv", "t"))
            .unwrap()
            .value;
        assert_eq!(z2.nrows(), 2);
        let h = HashingVectorizer::new(8)
            .transform_docs(&docs, &Session::new("hv", "t"))
            .unwrap()
            .value;
        assert_eq!(h.shape(), (2, 8));
        let recs = ["color=1 year=2012", "color=0 year=2013 extra=1"];
        let mut dv = DictVectorizer::new();
        dv.fit_docs(&recs, &Session::new("dv", "fit")).unwrap();
        assert!(dv.vocabulary().contains(&"color".to_string()));
        let xd = dv
            .transform_docs(&recs, &Session::new("dv", "t"))
            .unwrap()
            .value;
        assert_eq!(xd.nrows(), 2);
        assert!(xd.ncols() >= 2);
        assert!(xd.get(0, 0).is_finite());
    }
}
