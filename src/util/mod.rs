use jagua_rs::Instant;
use numfmt::{Formatter, Precision, Scales};
use std::sync::LazyLock;

pub mod assertions;

pub mod io;
pub mod listener;
pub mod solve;
pub mod svg_exporter;
pub mod terminator;

/// Reference point for the run times reported in exported solutions.
pub static EPOCH: LazyLock<Instant> = LazyLock::new(Instant::now);

pub(crate) static FMT: fn() -> Formatter = || -> Formatter {
    Formatter::new()
        .scales(Scales::short())
        .precision(Precision::Significance(3))
};
