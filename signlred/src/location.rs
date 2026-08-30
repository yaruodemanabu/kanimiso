//! Where in the computation an issue was raised.

use core::fmt;

/// Source location plus optional matrix / sample coordinates.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Location {
    /// Rust module path (`module_path!()`).
    pub module: &'static str,
    /// Source file.
    pub file: &'static str,
    /// Source line.
    pub line: u32,
    /// Optional named site (`"gram_matrix"`, `"emission[k]"`).
    pub site: Option<String>,
    /// Optional row index in the design / series.
    pub row: Option<usize>,
    /// Optional column / feature index.
    pub column: Option<usize>,
}

impl Location {
    /// Capture the caller's source location.
    pub fn here(module: &'static str, file: &'static str, line: u32) -> Self {
        Self {
            module,
            file,
            line,
            site: None,
            row: None,
            column: None,
        }
    }

    /// Attach a logical site name.
    pub fn with_site(mut self, site: impl Into<String>) -> Self {
        self.site = Some(site.into());
        self
    }

    /// Attach a row/column pair.
    pub fn at(mut self, row: usize, column: usize) -> Self {
        self.row = Some(row);
        self.column = Some(column);
        self
    }
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.module)?;
        if let Some(site) = &self.site {
            write!(f, " site={site}")?;
        }
        if let (Some(r), Some(c)) = (self.row, self.column) {
            write!(f, " cell=({r},{c})")?;
        }
        Ok(())
    }
}

/// Capture location at the call site.
#[macro_export]
macro_rules! here {
    () => {
        $crate::Location::here(module_path!(), file!(), line!())
    };
    ($site:expr) => {
        $crate::Location::here(module_path!(), file!(), line!()).with_site($site)
    };
}
