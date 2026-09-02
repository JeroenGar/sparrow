mod dashboard;
mod logging;
mod runtime;

use anyhow::{Result, bail};
use clap::Parser;
use dashboard::SearchBudget;
use jagua_rs::io::import::Importer;
use rand::SeedableRng;
use rand::rngs::Xoshiro256PlusPlus;
use sparrow::config::{DEFAULT_SPARROW_CONFIG, ShrinkDecayStrategy, SparrowConfig};
use sparrow::consts::{
    DEFAULT_COMPRESS_TIME_RATIO, DEFAULT_EXPLORE_TIME_RATIO, DEFAULT_FAIL_DECAY_RATIO_CMPR,
    DEFAULT_MAX_CONSEQ_FAILS_EXPL, LOG_LEVEL_FILTER_DEBUG, LOG_LEVEL_FILTER_RELEASE,
};
use sparrow::util::io::{self, MainCli};
use std::fs;
use std::path::Path;
use std::time::Duration;

pub(super) const OUTPUT_DIR: &str = "output";

pub fn run() -> Result<()> {
    let args = MainCli::parse();
    fs::create_dir_all(OUTPUT_DIR)?;
    let log_level = match cfg!(debug_assertions) {
        true => LOG_LEVEL_FILTER_DEBUG,
        false => LOG_LEVEL_FILTER_RELEASE,
    };
    let logs = logging::init(log_level, Path::new("output/log.txt"))?;

    let config = configure(&args)?;
    let budget = SearchBudget::from_config(&config);
    let rng = Xoshiro256PlusPlus::seed_from_u64(
        config
            .rng_seed
            .map_or_else(rand::random, |seed| seed as u64),
    );

    let (ext_instance, ext_solution) = io::read_spp_input(Path::new(&args.input))?;
    let importer = Importer::new(
        config.cde_config,
        config.poly_simpl_tolerance,
        config.min_item_separation,
        config.narrow_concavity_cutoff_ratio,
    );
    let instance = jagua_rs::probs::spp::io::import_instance(&importer, &ext_instance)?;
    let initial_solution = ext_solution
        .map(|solution| jagua_rs::probs::spp::io::import_solution(&instance, &solution));

    let solution = runtime::run(
        instance.clone(),
        &ext_instance,
        initial_solution,
        config,
        rng,
        logs,
        budget,
    )?;

    let svg_path = format!("{OUTPUT_DIR}/final_{}.svg", ext_instance.name);
    let json_path = format!("{OUTPUT_DIR}/final_{}.json", ext_instance.name);
    println!(
        "Finished at width {:.3}, density {:.3}%\n{svg_path}\n{json_path}",
        solution.strip_width(),
        solution.density(&instance) * 100.0,
    );
    Ok(())
}

fn configure(args: &MainCli) -> Result<SparrowConfig> {
    let mut config = DEFAULT_SPARROW_CONFIG;
    let (exploration, compression) = match (args.global_time, args.exploration, args.compression) {
        (Some(total), None, None) => (
            Duration::from_secs(total).mul_f32(DEFAULT_EXPLORE_TIME_RATIO),
            Duration::from_secs(total).mul_f32(DEFAULT_COMPRESS_TIME_RATIO),
        ),
        (None, Some(exploration), Some(compression)) => (
            Duration::from_secs(exploration),
            Duration::from_secs(compression),
        ),
        (None, None, None) => (
            Duration::from_secs(600).mul_f32(DEFAULT_EXPLORE_TIME_RATIO),
            Duration::from_secs(600).mul_f32(DEFAULT_COMPRESS_TIME_RATIO),
        ),
        _ => bail!("invalid time limit arguments"),
    };
    config.expl_cfg.time_limit = exploration;
    config.cmpr_cfg.time_limit = compression;
    if args.early_termination {
        config.expl_cfg.max_conseq_failed_attempts = Some(DEFAULT_MAX_CONSEQ_FAILS_EXPL);
        config.cmpr_cfg.shrink_decay =
            ShrinkDecayStrategy::FailureBased(DEFAULT_FAIL_DECAY_RATIO_CMPR);
    }
    args.apply_config_overrides(&mut config);
    Ok(config)
}
