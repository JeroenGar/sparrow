use crate::consts::{
    DEFAULT_COMPRESS_TIME_RATIO, DEFAULT_EXPLORE_TIME_RATIO, DEFAULT_FAIL_DECAY_RATIO_CMPR,
    DEFAULT_MAX_CONSEQ_FAILS_EXPL,
};
use crate::optimizer::separator::SeparatorConfig;
use crate::sample::search::SampleConfig;
use jagua_rs::collision_detection::CDEConfig;
use jagua_rs::geometry::fail_fast::SPSurrogateConfig;
use std::time::Duration;

/// Configuration of a complete sparrow run.
///
/// Start from [`SparrowConfig::default`] (the configuration used in the paper) and adjust it,
/// for example with [`SparrowConfig::with_time_limit`].
#[derive(Debug, Clone, Copy)]
pub struct SparrowConfig {
    /// Seed for the random number generator. `None` picks a random seed.
    pub rng_seed: Option<usize>,
    /// Configuration of the exploration phase.
    pub expl_cfg: ExplorationConfig,
    /// Configuration of the compression phase.
    pub cmpr_cfg: CompressionConfig,
    /// Configuration for the collision detection engine.
    /// See [`CDEConfig`] for more details.
    pub cde_config: CDEConfig,
    /// Defines the polygon simplification tolerance: maximum allowable inflation of items when simplifying their shape.
    /// Disabled if `None`.
    /// See [`jagua_rs::io::import::Importer::new`] for more details.
    pub poly_simpl_tolerance: Option<f32>,
    /// Defines a maximum distance and area of a concavity to be considered "narrow" (which will be closed).
    /// Disabled if `None`.
    /// See [`jagua_rs::io::import::Importer::new`] for more details.
    pub narrow_concavity_cutoff_ratio: Option<(f32, f32)>,
}

impl Default for SparrowConfig {
    fn default() -> Self {
        DEFAULT_SPARROW_CONFIG
    }
}

impl SparrowConfig {
    /// Splits a total time budget over the two phases:
    /// 80% to exploration and 20% to compression.
    pub fn with_time_limit(mut self, total: Duration) -> Self {
        self.expl_cfg.time_limit = total.mul_f32(DEFAULT_EXPLORE_TIME_RATIO);
        self.cmpr_cfg.time_limit = total.mul_f32(DEFAULT_COMPRESS_TIME_RATIO);
        self
    }

    /// Lets both phases end before their time limit once they stop making progress.
    ///
    /// Exploration ends after a number of consecutive failed attempts to find a feasible
    /// solution at a narrower strip width. Compression shrinks its step after every
    /// failure instead of over time, and ends once the step is smaller than the end of
    /// [`CompressionConfig::shrink_range`].
    pub fn with_early_termination(mut self) -> Self {
        self.expl_cfg.max_conseq_failed_attempts = Some(DEFAULT_MAX_CONSEQ_FAILS_EXPL);
        self.cmpr_cfg.shrink_decay = ShrinkDecayStrategy::FailureBased(DEFAULT_FAIL_DECAY_RATIO_CMPR);
        self
    }
}

/// Configuration of the exploration phase (Algorithm 12 in the paper).
///
/// Exploration repeatedly shrinks the strip and tries to resolve all overlap at the new width.
#[derive(Debug, Clone, Copy)]
pub struct ExplorationConfig {
    /// Fraction by which the strip width shrinks after each feasible solution (`0.001` is 0.1%).
    pub shrink_step: f32,
    /// Time budget of the phase.
    pub time_limit: Duration,
    /// Ends the phase after this many consecutive failed attempts at the same width.
    /// `None` keeps going until the time limit.
    pub max_conseq_failed_attempts: Option<usize>,
    /// Standard deviation of the normal distribution that picks which failed attempt to
    /// restart from. Smaller values favour the attempts with the least overlap.
    pub solution_pool_distribution_stddev: f32,
    /// Configuration of the separator, which resolves overlap at a fixed width.
    pub separator_config: SeparatorConfig,
    /// Items whose convex hulls together make up this fraction of the total convex hull area,
    /// largest first, count as "large". Disruptions swap two large items.
    pub large_item_ch_area_cutoff_percentile: f32
}

/// Configuration of the compression phase (Algorithm 13 in the paper).
///
/// Compression starts from the best exploration solution and tries ever smaller shrinks.
#[derive(Debug, Clone, Copy)]
pub struct CompressionConfig {
    /// Start and end of the shrink step, as fractions of the strip width.
    /// The step decays from the first value towards the second.
    pub shrink_range: (f32, f32),
    /// Time budget of the phase.
    pub time_limit: Duration,
    /// How the shrink step decays over the phase.
    pub shrink_decay: ShrinkDecayStrategy,
    /// Configuration of the separator, which resolves overlap at a fixed width.
    pub separator_config: SeparatorConfig,
}

/// How the compression phase decays its shrink step.
#[derive(Debug, Clone, Copy)]
pub enum ShrinkDecayStrategy {
    /// The shrink ratio decays linearly with time
    TimeBased,
    /// The shrink ratio decays by a fixed ratio every time it fails to compress into a feasible solution
    FailureBased(f32),
}

/// The configuration used in the paper, with 9 minutes of exploration and 1 minute of compression.
pub const DEFAULT_SPARROW_CONFIG: SparrowConfig = SparrowConfig {
    rng_seed: None,
    expl_cfg: ExplorationConfig {
        shrink_step: 0.001,
        time_limit: Duration::from_secs(9 * 60),
        max_conseq_failed_attempts: None,
        solution_pool_distribution_stddev: 0.25,
        separator_config: SeparatorConfig {
            iter_no_imprv_limit: 200,
            strike_limit: 3,
            log_level: log::Level::Info,
            n_workers: 3,
            sample_config: SampleConfig {
                n_container_samples: 50,
                n_focussed_samples: 25,
                n_coord_descents: 3,
            },
        },
        large_item_ch_area_cutoff_percentile: 0.75
    },
    cmpr_cfg: CompressionConfig {
        shrink_range: (0.0005, 0.00001),
        time_limit: Duration::from_secs(60),
        shrink_decay: ShrinkDecayStrategy::TimeBased,
        separator_config: SeparatorConfig {
            iter_no_imprv_limit: 100,
            strike_limit: 5,
            log_level: log::Level::Debug,
            n_workers: 3,
            sample_config: SampleConfig {
                n_container_samples: 50,
                n_focussed_samples: 25,
                n_coord_descents: 3,
            },
        },
    },
    cde_config: CDEConfig {
        quadtree_depth: 4,
        cd_threshold: 64,
        item_surrogate_config: SPSurrogateConfig {
            n_pole_limits: [(64, 0.0), (16, 0.8), (8, 0.9)],
            ff_pole_area_ratio: 0.5,
            n_ff_piers: 0,
        },
    },
    poly_simpl_tolerance: Some(0.001),
    narrow_concavity_cutoff_ratio: Some((0.01, 0.01)),
};
