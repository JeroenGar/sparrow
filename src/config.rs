use crate::optimizer::separator::SeparatorConfig;
use crate::sample::search::SampleConfig;
use crate::util::io::svg_util::{SvgDrawOptions, SvgLayoutTheme};
use jagua_rs::util::config::{CDEConfig, SPSurrogateConfig};
use jagua_rs::util::polygon_simplification::PolySimplConfig;

pub const RNG_SEED: Option<usize> = None;

pub const OUTPUT_DIR: &str = "output";

pub const LIVE_DIR: &str = "assets/live";

pub const LOG_LEVEL_FILTER_RELEASE: log::LevelFilter = log::LevelFilter::Info;

pub const LOG_LEVEL_FILTER_DEBUG: log::LevelFilter = log::LevelFilter::Debug;

pub const DRAW_OPTIONS: SvgDrawOptions = SvgDrawOptions {
    theme: SvgLayoutTheme::GRAY_THEME,
    quadtree: false,
    haz_prox_grid: false,
    surrogate: false,
    highlight_overlap: true,
};

pub const CDE_CONFIG: CDEConfig = CDEConfig {
    quadtree_depth: 4,
    hpg_n_cells: 0,
    item_surrogate_config: SPSurrogateConfig {
        pole_coverage_goal: 0.9,
        max_poles: 20,
        n_ff_poles: 2,
        n_ff_piers: 0,
    },
};

pub const SIMPLIFICATION_CONFIG: PolySimplConfig = PolySimplConfig::Disabled;

pub const LBF_SAMPLE_CONFIG: SampleConfig = SampleConfig {
    n_bin_samples: 1000,
    n_focussed_samples: 0,
    n_coord_descents: 3,
};

pub const EXPLORE_SOL_DISTR_STDDEV: f32 = 0.25;
pub const EXPLORE_R_SHRINK: f32 = 0.005;

pub const SEP_CONFIG_EXPLORE: SeparatorConfig = SeparatorConfig {
    iter_no_imprv_limit: 100,
    strike_limit: 5,
    log_level: log::Level::Info,
    n_workers: 2,
    sample_config: SampleConfig {
        n_bin_samples: 50,
        n_focussed_samples: 25,
        n_coord_descents: 3,
    }
};

pub const LARGE_AREA_CH_AREA_CUTOFF_RATIO: f32 = 0.5;
pub const WEIGHT_MAX_INC_RATIO: f32 = 2.0;
pub const WEIGHT_MIN_INC_RATIO: f32 = 1.35;
pub const WEIGHT_OVERLAP_DECAY: f32 = 0.95;
pub const OVERLAP_PROXY_EPSILON_DIAM_RATIO: f32 = 0.01;
pub const COMPRESS_R_SHRINKS: [f32; 2] = [0.0005, 0.0001];
pub const COMPRESS_N_STRIKES: [usize; 2] = [5,5];

pub const SEPARATOR_CONFIG_COMPRESS: SeparatorConfig = SeparatorConfig {
    iter_no_imprv_limit: 100,
    strike_limit: 5,
    log_level: log::Level::Debug,
    n_workers: 2,
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