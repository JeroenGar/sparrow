use jagua_rs::collision_detection::CDEConfig;
use jagua_rs::geometry::fail_fast::SPSurrogateConfig;
use crate::optimizer::separator::SeparatorConfig;
use crate::sample::search::SampleConfig;
use crate::util::io::svg_util::{SvgDrawOptions, SvgLayoutTheme};

pub const RNG_SEED: Option<usize> = None;

pub const CDE_CONFIG: CDEConfig = CDEConfig {
    quadtree_depth: 4,
    item_surrogate_config: SPSurrogateConfig {
        n_pole_limits: [(64, 0.0), (16, 0.8), (8, 0.9)],
        n_ff_poles: 2,
        n_ff_piers: 0,
    },
};

/// Defines the polygon simplification tolerance: maximum allowable inflation of items when simplifying their shape.
/// Disabled if `None`.
/// See [`jagua_rs::io::parser::Parser::new`] for more details.
pub const SIMPL_TOLERANCE: Option<f32> = Some(0.001);


/// Defines the minimum distance between items and other hazards.
/// Disabled if `None`.
/// See [`jagua_rs::io::parser::Parser::new`] for more details.
pub const MIN_ITEM_SEPARATION: Option<f32> = None;

pub const EXPLORE_SHRINK_STEP: f32 = 0.001;
pub const EXPLORE_SOL_DISTR_STDDEV: f32 = 0.25;
pub const EXPLORE_TIME_RATIO: f32 = 0.8;

pub const COMPRESS_SHRINK_RANGE: (f32, f32) = (0.0005, 0.00001);
pub const COMPRESS_TIME_RATIO: f32 = 1.0 - EXPLORE_TIME_RATIO;

pub const WEIGHT_MAX_INC_RATIO: f32 = 2.0;
pub const WEIGHT_MIN_INC_RATIO: f32 = 1.2;
pub const WEIGHT_DECAY: f32 = 0.95;

pub const OVERLAP_PROXY_EPSILON_DIAM_RATIO: f32 = 0.01;

pub const SEP_CFG_EXPLORE: SeparatorConfig = SeparatorConfig {
    iter_no_imprv_limit: 200,
    strike_limit: 3,
    log_level: log::Level::Info,
    n_workers: 3,
    sample_config: SampleConfig {
        n_bin_samples: 50,
        n_focussed_samples: 25,
        n_coord_descents: 3,
    }
};

pub const SEP_CFG_COMPRESS: SeparatorConfig = SeparatorConfig {
    iter_no_imprv_limit: 100,
    strike_limit: 5,
    log_level: log::Level::Debug,
    n_workers: 3,
    sample_config: SampleConfig {
        n_bin_samples: 50,
        n_focussed_samples: 25,
        n_coord_descents: 3,
    },
};

/// Coordinate descent step multiplier on success
pub const CD_STEP_SUCCESS: f32 = 1.1;

/// Coordinate descent step multiplier on failure
pub const CD_STEP_FAIL: f32 = 0.5;

/// Ratio of the item's min dimension to be used as initial and limit step size for the first refinement
pub const PRE_REF_CD_RATIOS: (f32, f32) = (0.25, 0.02);

/// Ratio of the item's min dimension to be used as initial and limit step size for the final refinement
pub const FIN_REF_CD_RATIOS: (f32, f32) = (0.01, 0.001);

/// If two samples are closer than this ratio of the item's min dimension, they are considered duplicates
pub const UNIQUE_SAMPLE_THRESHOLD: f32 = 0.05;

pub const OUTPUT_DIR: &str = "output";

pub const LIVE_DIR: &str = "data/live";

pub const LOG_LEVEL_FILTER_RELEASE: log::LevelFilter = log::LevelFilter::Info;

pub const LOG_LEVEL_FILTER_DEBUG: log::LevelFilter = log::LevelFilter::Debug;

pub const LARGE_AREA_CH_AREA_CUTOFF_RATIO: f32 = 0.5;

pub const DRAW_OPTIONS: SvgDrawOptions = SvgDrawOptions {
    theme: SvgLayoutTheme::GRAY_THEME,
    quadtree: false,
    surrogate: false,
    highlight_collisions: true,
    draw_cd_shapes: false,
    highlight_cd_shapes: true,
};

pub const LBF_SAMPLE_CONFIG: SampleConfig = SampleConfig {
    n_bin_samples: 1000,
    n_focussed_samples: 0,
    n_coord_descents: 3,
};