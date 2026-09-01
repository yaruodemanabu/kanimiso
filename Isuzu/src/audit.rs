//! Compile-time / test-time Pure-Rust dependency policy.

/// Direct dependencies declared in this crate's `Cargo.toml`.
/// Every crate in this list is Pure Rust (no C/Fortran/BLAS bindings).
pub const DIRECT_DEPS: &[&str] = &["amatsuki", "faer", "rustfft", "thiserror"];

/// Crate names that pull native / BLAS / LAPACK / OpenSSL code.
/// A test fails if any of these appear in `Cargo.lock`.
pub const FORBIDDEN_NATIVE: &[&str] = &[
    "openblas-src",
    "netlib-src",
    "intel-mkl-src",
    "blas-src",
    "lapack-src",
    "lapack",
    "blas",
    "cblas",
    "accelerate-src",
    "ndarray-linalg",
    "nalgebra-lapack",
    "openssl",
    "openssl-sys",
    "libsqlite3-sys",
    "bindgen",
    "cc",
    "cmake",
    "gsl",
    "rgsl",
    "statrs-gsl",
    "libcblas",
    "openblas",
    "mkl-sys",
    "suitesparse-sys",
    "arpack-sys",
];

/// Parse `Cargo.lock` and return package names.
pub fn lockfile_packages(lock: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in lock.lines() {
        if let Some(rest) = line.strip_prefix("name = \"") {
            if let Some(name) = rest.strip_suffix('"') {
                out.push(name.to_string());
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cargo_lock_has_no_forbidden_native_crates() {
        let lock = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.lock"));
        let pkgs = lockfile_packages(lock);
        assert!(
            !pkgs.is_empty(),
            "Cargo.lock is empty; run cargo generate-lockfile"
        );
        let mut hits = Vec::new();
        for p in &pkgs {
            if FORBIDDEN_NATIVE.contains(&p.as_str()) {
                hits.push(p.clone());
            }
        }
        assert!(
            hits.is_empty(),
            "forbidden native/BLAS crates in Cargo.lock: {hits:?}"
        );
    }

    #[test]
    fn direct_deps_are_the_declared_pure_rust_set() {
        let manifest = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"));
        for dep in DIRECT_DEPS {
            assert!(
                manifest.contains(dep),
                "expected direct dependency {dep} in Cargo.toml"
            );
        }
        assert!(
            !manifest.contains("nalgebra"),
            "nalgebra must not remain a direct dependency"
        );
        assert!(
            !manifest.contains("rand_chacha"),
            "rand_* crates must be replaced by amatsuki"
        );
    }

    #[test]
    fn lockfile_has_no_nalgebra_or_rand_stack() {
        let lock = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.lock"));
        let pkgs = lockfile_packages(lock);
        for banned in ["nalgebra", "rand", "rand_chacha", "rand_distr"] {
            assert!(
                !pkgs.iter().any(|p| p == banned),
                "{banned} must not appear in Cargo.lock"
            );
        }
        assert!(
            pkgs.iter().any(|p| p == "faer"),
            "faer must appear in Cargo.lock"
        );
        assert!(
            pkgs.iter().any(|p| p == "amatsuki"),
            "amatsuki must appear in Cargo.lock"
        );
    }
}
