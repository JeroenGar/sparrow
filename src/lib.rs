#![doc = include_str!("../README.crates.md")]
#![cfg_attr(feature = "simd", feature(portable_simd))]
#![allow(const_item_mutation)]
#![allow(unused_imports)]

use crate::config::SparrowConfig;
use crate::optimizer::optimize;
use crate::util::listener::DummySolListener;
use crate::util::terminator::BasicTerminator;
use jagua_rs::Instant;
use jagua_rs::io::import::Importer;
use jagua_rs::io::svg::s_layout_to_svg;
use jagua_rs::probs::spp::io::{export, import_instance, import_solution};
use numfmt::{Formatter, Precision, Scales};
use rand::SeedableRng;
use rand::rngs::Xoshiro256PlusPlus;
use anyhow::Context;
use std::path::Path;
use std::sync::LazyLock;

pub use jagua_rs::probs::spp::io::ext_repr::{ExtSPInstance, ExtSPSolution};

pub mod optimizer;
pub mod quantify;
pub mod sample;
pub mod util;
pub mod config;
pub mod eval;
pub mod consts;

/// Reads an instance from a JSON file in the `jagua-rs` format.
///
/// Also accepts a solution file (see [`read_solution`]); its solution part is ignored.
pub fn read_instance(path: impl AsRef<Path>) -> anyhow::Result<ExtSPInstance> {
    let path = path.as_ref();
    parse_instance(&read_file(path)?).with_context(|| format!("could not read {}", path.display()))
}

/// Reads a solution file: an instance together with a solution for it, as written by
/// [`write_solution`] and the sparrow CLI (`output/final_{name}.json`).
///
/// Use it with [`solve_from`] to continue from an earlier result. Returns an error if the file
/// contains only an instance.
pub fn read_solution(path: impl AsRef<Path>) -> anyhow::Result<(ExtSPInstance, ExtSPSolution)> {
    let path = path.as_ref();
    parse_solution(&read_file(path)?).with_context(|| format!("could not read {}", path.display()))
}

/// Writes a solution file: `instance` together with `solution`, as produced by [`to_json`].
/// Creates missing parent directories.
pub fn write_solution(path: impl AsRef<Path>, instance: &ExtSPInstance, solution: &ExtSPSolution) -> anyhow::Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, to_json(instance, solution)?)
        .with_context(|| format!("could not write {}", path.display()))
}

/// Parses an instance from JSON in the `jagua-rs` format.
/// The in-memory counterpart of [`read_instance`].
///
/// Also accepts a solution (see [`parse_solution`]); its solution part is ignored.
pub fn parse_instance(json: &str) -> anyhow::Result<ExtSPInstance> {
    let (instance, _) = util::io::parse_spp_input(json)?;
    Ok(instance)
}

/// Parses an instance together with a solution for it, as produced by [`to_json`].
/// The in-memory counterpart of [`read_solution`].
///
/// Returns an error if the JSON contains only an instance.
pub fn parse_solution(json: &str) -> anyhow::Result<(ExtSPInstance, ExtSPSolution)> {
    match util::io::parse_spp_input(json)? {
        (instance, Some(solution)) => Ok((instance, solution)),
        (_, None) => anyhow::bail!("found an instance but no solution"),
    }
}

/// Serializes `instance` together with `solution` as JSON, the format of the solution files
/// that the sparrow CLI writes. The in-memory counterpart of [`write_solution`].
pub fn to_json(instance: &ExtSPInstance, solution: &ExtSPSolution) -> anyhow::Result<String> {
    #[derive(serde::Serialize)]
    struct SolutionFile<'a> {
        #[serde(flatten)]
        instance: &'a ExtSPInstance,
        solution: &'a ExtSPSolution,
    }

    Ok(serde_json::to_string_pretty(&SolutionFile { instance, solution })?)
}

fn read_file(path: &Path) -> anyhow::Result<String> {
    std::fs::read_to_string(path).with_context(|| format!("could not read {}", path.display()))
}

/// Draws `solution` as an SVG document, returned as a string.
///
/// The SVG is both a picture and an exact record of the solution: it defines every item's
/// original shape and the transformation applied to it.
///
/// Returns an error if `solution` does not fit `instance`.
pub fn to_svg(instance: &ExtSPInstance, solution: &ExtSPSolution) -> anyhow::Result<String> {
    let config = SparrowConfig::default();
    let importer = Importer::new(config.cde_config, config.poly_simpl_tolerance, config.narrow_concavity_cutoff_ratio);
    let solution = import_solution(&import_instance(&importer, instance)?, solution)?;
    let svg = s_layout_to_svg(solution.layout_snapshot(), 0, consts::DRAW_OPTIONS, &instance.name);
    Ok(svg.to_string())
}

/// Nests all items of `instance` in a strip of minimal width.
///
/// Runs the complete heuristic (construction, exploration and compression) with `config`,
/// and returns the best feasible solution found. The run takes as long as the time limits in
/// `config` allow, unless [`SparrowConfig::with_early_termination`] lets it stop sooner.
///
/// Returns an error if `instance` is invalid, or if no initial solution can be constructed.
pub fn solve(instance: &ExtSPInstance, config: &SparrowConfig) -> anyhow::Result<ExtSPSolution> {
    run(instance, None, config)
}

/// Like [`solve`], but starts from `initial` instead of constructing a first solution.
///
/// `initial` should be a feasible solution for `instance` that places every item, such as the
/// result of an earlier run: sparrow treats it as its first feasible solution and searches for
/// narrower ones from there.
///
/// Returns an error if `instance` is invalid, or if `initial` does not fit it.
pub fn solve_from(instance: &ExtSPInstance, initial: &ExtSPSolution, config: &SparrowConfig) -> anyhow::Result<ExtSPSolution> {
    run(instance, Some(initial), config)
}

fn run(instance: &ExtSPInstance, initial: Option<&ExtSPSolution>, config: &SparrowConfig) -> anyhow::Result<ExtSPSolution> {
    let start = Instant::now();
    let importer = Importer::new(config.cde_config, config.poly_simpl_tolerance, config.narrow_concavity_cutoff_ratio);
    let instance = import_instance(&importer, instance)?;
    let initial = initial.map(|solution| import_solution(&instance, solution)).transpose()?;
    let seed = config.rng_seed.map_or_else(rand::random, |seed| seed as u64);

    let solution = optimize(
        instance,
        Xoshiro256PlusPlus::seed_from_u64(seed),
        &mut DummySolListener,
        &mut BasicTerminator::new(),
        &config.expl_cfg,
        &config.cmpr_cfg,
        initial.as_ref(),
    )?;
    Ok(export(&solution, start))
}

/// Reference point for the run times reported in exported solutions.
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