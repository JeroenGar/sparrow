use crate::config::SparrowConfig;
use crate::consts::DRAW_OPTIONS;
use anyhow::{Context, Result};
use jagua_rs::io::import::Importer;
use jagua_rs::io::svg::s_layout_to_svg;
use jagua_rs::probs::spp::io::{import_instance, import_solution};
use jagua_rs::probs::spp::io::ext_repr::{ExtSPInstance, ExtSPSolution};
use log::{log, Level};
use serde::{Deserialize, Serialize};
use std::fs;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use svg::Document;

#[derive(Serialize, Deserialize, Clone)]
pub struct ExtSPOutput {
    #[serde(flatten)]
    pub instance: ExtSPInstance,
    pub solution: ExtSPSolution,
}

pub fn write_svg(document: &Document, path: &Path, log_lvl: Level) -> Result<()> {
    //make sure the parent directory exists
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).context("could not create parent directory for svg file")?;
    }
    svg::save(path, document)?;
    log!(log_lvl,
        "[IO] svg exported to file://{}",
        fs::canonicalize(path)
            .expect("could not canonicalize path")
            .to_str()
            .unwrap()
    );
    Ok(())
}

pub fn write_json(json: &impl Serialize, path: &Path, log_lvl: Level) -> Result<()> {
    let file = File::create(path)?;
    serde_json::to_writer_pretty(file, json)?;
    log!(log_lvl,
        "[IO] json exported to file://{}",
        fs::canonicalize(path)
            .expect("could not canonicalize path")
            .to_str()
            .unwrap()
    );
    Ok(())
}

pub fn read_spp_input(path: &Path) -> Result<(ExtSPInstance, Option<ExtSPSolution>)> {
    let input_str = fs::read_to_string(path).context("could not read input file")?;
    parse_spp_input(&input_str)
}

/// Parses JSON holding either an instance, or an instance together with a solution.
pub fn parse_spp_input(json: &str) -> Result<(ExtSPInstance, Option<ExtSPSolution>)> {
    //try parsing a full output (instance + solution)
    match serde_json::from_str::<ExtSPOutput>(json) {
        Ok(ext_output) => {
            Ok((ext_output.instance, Some(ext_output.solution)))
        }
        Err(_) => {
            //try parsing just the instance
            let ext_instance = serde_json::from_str::<ExtSPInstance>(json)
                .context("could not parse instance")?;
            Ok((ext_instance, None))
        }
    }
}

/// Reads an instance from a JSON file in the `jagua-rs` format.
///
/// Also accepts a solution file (see [`read_solution`]); its solution part is ignored.
pub fn read_instance(path: impl AsRef<Path>) -> Result<ExtSPInstance> {
    let path = path.as_ref();
    parse_instance(&read_file(path)?).with_context(|| format!("could not read {}", path.display()))
}

/// Reads a solution file: an instance together with a solution for it, as written by
/// [`write_solution`] and the sparrow CLI (`output/final_{name}.json`).
///
/// Use it with [`solve_from`](crate::solve_from) to continue from an earlier result.
/// Returns an error if the file contains only an instance.
pub fn read_solution(path: impl AsRef<Path>) -> Result<(ExtSPInstance, ExtSPSolution)> {
    let path = path.as_ref();
    parse_solution(&read_file(path)?).with_context(|| format!("could not read {}", path.display()))
}

/// Writes a solution file: `instance` together with `solution`, as produced by [`to_json`].
/// Creates missing parent directories.
pub fn write_solution(path: impl AsRef<Path>, instance: &ExtSPInstance, solution: &ExtSPSolution) -> Result<()> {
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
pub fn parse_instance(json: &str) -> Result<ExtSPInstance> {
    let (instance, _) = parse_spp_input(json)?;
    Ok(instance)
}

/// Parses an instance together with a solution for it, as produced by [`to_json`].
/// The in-memory counterpart of [`read_solution`].
///
/// Returns an error if the JSON contains only an instance.
pub fn parse_solution(json: &str) -> Result<(ExtSPInstance, ExtSPSolution)> {
    match parse_spp_input(json)? {
        (instance, Some(solution)) => Ok((instance, solution)),
        (_, None) => anyhow::bail!("found an instance but no solution"),
    }
}

/// Serializes `instance` together with `solution` as JSON, the format of the solution files
/// that the sparrow CLI writes. The in-memory counterpart of [`write_solution`].
pub fn to_json(instance: &ExtSPInstance, solution: &ExtSPSolution) -> Result<String> {
    #[derive(serde::Serialize)]
    struct SolutionFile<'a> {
        #[serde(flatten)]
        instance: &'a ExtSPInstance,
        solution: &'a ExtSPSolution,
    }

    Ok(serde_json::to_string_pretty(&SolutionFile { instance, solution })?)
}

fn read_file(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).with_context(|| format!("could not read {}", path.display()))
}

/// Draws `solution` as an SVG document, returned as a string.
///
/// The SVG is both a picture and an exact record of the solution: it defines every item's
/// original shape and the transformation applied to it.
///
/// Returns an error if `solution` does not fit `instance`.
pub fn to_svg(instance: &ExtSPInstance, solution: &ExtSPSolution) -> Result<String> {
    let config = SparrowConfig::default();
    let importer = Importer::new(config.cde_config, config.poly_simpl_tolerance, config.narrow_concavity_cutoff_ratio);
    let solution = import_solution(&import_instance(&importer, instance)?, solution)?;
    let svg = s_layout_to_svg(solution.layout_snapshot(), 0, DRAW_OPTIONS, &instance.name);
    Ok(svg.to_string())
}
