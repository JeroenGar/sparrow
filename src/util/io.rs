use crate::EPOCH;
use crate::config::SparrowConfig;
use anyhow::{Context, Result};
use clap::Parser;
use jagua_rs::probs::spp::io::ext_repr::{ExtSPInstance, ExtSPSolution};
use log::{log, warn, Level, LevelFilter};
use serde::{Deserialize, Serialize};
use std::fs;
use std::fs::File;
use std::io::BufReader;
use std::num::NonZeroUsize;
use std::path::Path;
use svg::Document;

#[derive(Parser)]
pub struct MainCli {
    /// Path to input file (mandatory)
    #[arg(short = 'i', long, help = "Path to the input JSON file, or a solution JSON file for warm starting")]
    pub input: String,

    /// Global time limit in seconds (mutually exclusive with -e and -c)
    #[arg(short = 't', long, conflicts_with_all = &["exploration", "compression"], help = "Set a global time limit (in seconds)")]
    pub global_time: Option<u64>,

    /// Exploration time limit in seconds (requires compression time)
    #[arg(short = 'e', long, requires = "compression", help = "Set the exploration phase time limit (in seconds)")]
    pub exploration: Option<u64>,

    /// Compression time limit in seconds (requires exploration time)
    #[arg(short = 'c', long, requires = "exploration", help = "Set the compression phase time limit (in seconds)")]
    pub compression: Option<u64>,

    /// Enable early and automatic termination
    #[arg(short = 'x', long, help = "Enable early termination of the optimization process")]
    pub early_termination: bool,

    #[arg(short = 's', long, help = "Fixed seed for the random number generator")]
    pub rng_seed: Option<u64>,

    /// Minimum distance between items and other hazards
    #[arg(long)]
    pub min_item_separation: Option<f32>,

    /// Number of worker threads used by the separator
    #[arg(long)]
    pub workers: Option<NonZeroUsize>,
}

impl MainCli {
    /// Applies optional CLI values that override the default optimizer configuration.
    pub fn apply_config_overrides(&self, config: &mut SparrowConfig) {
        if let Some(rng_seed) = self.rng_seed {
            warn!("[MAIN] overriding RNG seed: {rng_seed}");
            config.rng_seed = Some(rng_seed as usize);
        }
        if let Some(min_item_separation) = self.min_item_separation {
            warn!("[MAIN] overriding minimum item separation: {min_item_separation}");
            config.min_item_separation = Some(min_item_separation);
        }
        if let Some(workers) = self.workers {
            warn!("[MAIN] overriding separator worker count: {workers}");
            config.expl_cfg.separator_config.n_workers = workers.get();
            config.cmpr_cfg.separator_config.n_workers = workers.get();
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ExtSPOutput {
    #[serde(flatten)]
    pub instance: ExtSPInstance,
    pub solution: ExtSPSolution,
}

pub fn init_logger(level_filter: LevelFilter, log_file_path: &Path) -> Result<()> {
    //remove old log file
    let _ = fs::remove_file(log_file_path);
    fern::Dispatch::new()
        // Perform allocation-free log formatting
        .format(|out, message, record| {
            let handle = std::thread::current();
            let thread_name = handle.name().unwrap_or("-");

            let duration = EPOCH.elapsed();
            let sec = duration.as_secs() % 60;
            let min = (duration.as_secs() / 60) % 60;
            let hours = (duration.as_secs() / 60) / 60;

            let prefix = format!(
                "[{}] [{:0>2}:{:0>2}:{:0>2}] <{}>",
                record.level(),
                hours,
                min,
                sec,
                thread_name,
            );

            out.finish(format_args!("{:<25}{}", prefix, message))
        })
        // Add blanket level filter -
        .level(level_filter)
        .chain(std::io::stdout())
        .chain(fern::log_file(log_file_path)?)
        .apply()?;
    log!(
        Level::Info,
        "[EPOCH]: {}",
        jiff::Timestamp::now()
    );
    Ok(())
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
    //try parsing a full output (instance + solution)
    match serde_json::from_str::<ExtSPOutput>(&input_str) {
        Ok(ext_output) => {
            Ok((ext_output.instance, Some(ext_output.solution)))
        }
        Err(_) => {
            //try parsing just the instance
            let ext_instance = serde_json::from_str::<ExtSPInstance>(&input_str)
                .context("could not parse instance from input file")?;
            Ok((ext_instance, None))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::MainCli;
    use crate::config::DEFAULT_SPARROW_CONFIG;
    use clap::Parser;

    #[test]
    fn applies_runtime_configuration_overrides() {
        let args = MainCli::try_parse_from([
            "sparrow",
            "-i",
            "input.json",
            "--rng-seed",
            "42",
            "--min-item-separation=-1.25",
            "--workers",
            "2",
        ])
        .unwrap();
        let mut config = DEFAULT_SPARROW_CONFIG;

        args.apply_config_overrides(&mut config);

        assert_eq!(config.rng_seed, Some(42));
        assert_eq!(config.min_item_separation, Some(-1.25));
        assert_eq!(config.expl_cfg.separator_config.n_workers, 2);
        assert_eq!(config.cmpr_cfg.separator_config.n_workers, 2);
    }

    #[test]
    fn rejects_zero_workers() {
        assert!(
            MainCli::try_parse_from(["sparrow", "-i", "input.json", "--workers", "0"]).is_err()
        );
    }
}
