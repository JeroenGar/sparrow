#![cfg_attr(rustfmt, rustfmt::skip)]
#![cfg_attr(feature = "simd", feature(portable_simd))]
#![allow(const_item_mutation)]
#![allow(unused_imports)]

use jagua_rs::Instant;
use numfmt::{Formatter, Precision, Scales};
use std::sync::LazyLock;

#[rustfmt::skip]
pub mod optimizer;
#[rustfmt::skip]
pub mod quantify;
#[rustfmt::skip]
pub mod sample;
#[rustfmt::skip]
pub mod util;
#[rustfmt::skip]
pub mod config;
#[rustfmt::skip]
pub mod eval;
#[rustfmt::skip]
pub mod consts;

pub static EPOCH: LazyLock<Instant> = LazyLock::new(Instant::now);

static FMT: fn() -> Formatter = || -> Formatter {
    Formatter::new()
        .scales(Scales::short())
        .precision(Precision::Significance(3))
};


#[cfg(feature = "live_svg")]
pub const EXPORT_LIVE_SVG: bool = true;

#[cfg(not(feature = "live_svg"))]
pub const EXPORT_LIVE_SVG: bool = false;

#[cfg(feature = "only_final_svg")]
pub const EXPORT_ONLY_FINAL_SVG: bool = true;

#[cfg(not(feature = "only_final_svg"))]
pub const EXPORT_ONLY_FINAL_SVG: bool = false;

#[cfg(all(feature = "live_svg", feature = "only_final_svg"))]
compile_error!("The features `live_svg` and `only_final_svg` are mutually exclusive.");
