use crate::sample::search::SearchConfig;
use crate::util::io::svg_util::{SvgDrawOptions, SvgLayoutTheme};
use jagua_rs::fsize;
use jagua_rs::util::config::{CDEConfig, SPSurrogateConfig};

pub const OUTPUT_DIR: &str = "output";

pub const LOG_LEVEL_RELEASE: log::LevelFilter = log::LevelFilter::Info;

pub const LOG_LEVEL_DEBUG: log::LevelFilter = log::LevelFilter::Debug;

pub const RNG_SEED: Option<usize> = None;

pub const DRAW_OPTIONS: SvgDrawOptions = SvgDrawOptions {
    theme: SvgLayoutTheme::GRAY_THEME,
    quadtree: false,
    haz_prox_grid: false,
    surrogate: false,
    highlight_overlap: true,
};

pub const CDE_CONFIG: CDEConfig = CDEConfig {
    quadtree_depth: 3,
    hpg_n_cells: 0,
    item_surrogate_config: SPSurrogateConfig {
        pole_coverage_goal: 0.95,
        max_poles: 20,
        n_ff_poles: 4,
        n_ff_piers: 0,
    },
};

pub const CONSTR_SEARCH_CONFIG: SearchConfig = SearchConfig {
    n_bin_samples: 1000,
    n_focussed_samples: 0,
    n_coord_descents: 3,
};

pub const N_ITER_NO_IMPROVEMENT: usize = 50;
pub const N_STRIKES: usize = 5;
pub const N_BIN_SAMPLES: usize = 50;
pub const N_FOCUSSED_SAMPLES: usize = 25;
pub const N_COORD_DESCENTS: usize = 3;
pub const JUMP_COOLDOWN: usize = 5;
pub const N_WORKERS: usize = 2;
pub const OT_MAX_INCREASE: fsize = 2.0;
pub const OT_MIN_INCREASE: fsize = 1.2;
pub const OT_DECAY: fsize = 0.95;
pub const PROXY_EPSILON_DIAM_FRAC: fsize = 0.01;
pub const STDDEV_SPREAD: fsize = 4.0;
pub const LARGE_ITEM_CH_AREA_CUTOFF_RATIO: fsize = 0.5;
pub const R_SHRINK: fsize = 0.005;
pub const POST_R_SHRINKS: [fsize; 2] = [0.0005, 0.0001];
pub const POST_N_STRIKES: [usize; 2] = [5, 5];
