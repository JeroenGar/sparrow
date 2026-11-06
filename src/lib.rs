#![doc = include_str!("../README.crates.md")]
#![cfg_attr(feature = "simd", feature(portable_simd))]
#![allow(const_item_mutation)]
#![allow(unused_imports)]

pub mod optimizer;
pub mod quantify;
pub mod sample;
pub mod util;
pub mod config;
pub mod eval;
pub mod consts;

pub use jagua_rs::probs::spp::io::ext_repr::{ExtSPInstance, ExtSPSolution};
pub use util::io::{parse_instance, parse_solution, read_instance, read_solution, to_json, to_svg, write_solution};
pub use util::solve::{solve, solve_from};
pub use util::EPOCH;
pub(crate) use util::FMT;
