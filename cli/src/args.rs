use clap::Parser;
use log::warn;
use sparrow::config::SparrowConfig;
use std::num::NonZeroUsize;

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
        if let Some(workers) = self.workers {
            warn!("[MAIN] overriding separator worker count: {workers}");
            config.expl_cfg.separator_config.n_workers = workers.get();
            config.cmpr_cfg.separator_config.n_workers = workers.get();
        }
    }
}
