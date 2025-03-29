use crate::optimizer::separator::SeparatorConfig;
use crate::sample::search::SampleConfig;
use crate::util::io::svg_util::{SvgDrawOptions, SvgLayoutTheme};
use jagua_rs::util::config::{CDEConfig, SPSurrogateConfig};
use jagua_rs::util::polygon_simplification::PolySimplConfig;

pub const RNG_SEED: Option<usize> = None;

pub const CDE_CONFIG: CDEConfig = CDEConfig {
    quadtree_depth: 4,
    hpg_n_cells: 0,
    item_surrogate_config: SPSurrogateConfig {
        n_pole_limits: [(64, 0.0), (16, 0.8), (8, 0.9)],
        //n_pole_limits: [(10, 0.8), (5, 0.9), (2, 0.95)],
        n_ff_poles: 2,
        n_ff_piers: 0,
    },
};

// Uncomment if your instance contains polygons with a lot of vertices
// pub const SIMPLIFICATION_CONFIG: PolySimplConfig = PolySimplConfig::Enabled{
//     tolerance: 0.001
// };
pub const SIMPLIFICATION_CONFIG: PolySimplConfig = PolySimplConfig::Disabled;

pub const EXPLORE_SHRINK_STEP: f32 = 0.001;
pub const EXPLORE_SOL_DISTR_STDDEV: f32 = 0.25;
pub const EXPLORE_TIME_RATIO: f32 = 0.8;

pub const COMPRESS_SHRINK_RANGE: (f32, f32) = (0.0005, 0.00001);
pub const COMPRESS_TIME_RATIO: f32 = 1.0 - EXPLORE_TIME_RATIO;

pub const WEIGHT_MAX_INC_RATIO: f32 = 2.0;
pub const WEIGHT_MIN_INC_RATIO: f32 = 1.2;
pub const WEIGHT_OVERLAP_DECAY: f32 = 0.95;

pub const OVERLAP_PROXY_EPSILON_DIAM_RATIO: f32 = 0.01;

pub const SEP_CFG_EXPLORE: SeparatorConfig = SeparatorConfig {
    iter_no_imprv_limit: 200,
    strike_limit: 3,
    log_level: log::Level::Info,
    n_workers: 1,
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
    n_workers: 1,
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

/// Coordinate descent initial step size as a ratio of the item's min dimension
pub const CD_STEP_INIT_RATIO: f32 = 0.25; //25%

/// Coordinate descent step limit as a ratio of the item's min dimension
pub const CD_STEP_LIMIT_RATIO: f32 = 0.001; //0.1%

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
    haz_prox_grid: false,
    surrogate: false,
    highlight_overlap: true,
};

pub const LBF_SAMPLE_CONFIG: SampleConfig = SampleConfig {
    n_bin_samples: 1000,
    n_focussed_samples: 0,
    n_coord_descents: 3,
};