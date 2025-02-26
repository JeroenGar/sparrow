extern crate core;

use std::ops::Add;
use gls_strip_packing::{DRAW_OPTIONS, SVG_OUTPUT_DIR};
use gls_strip_packing::opt::constr_builder::ConstructiveBuilder;
use gls_strip_packing::opt::gls_orchestrator::{GLSOrchestrator, N_WORKERS};
use gls_strip_packing::sample::search::SearchConfig;
use gls_strip_packing::util::io;
use jagua_rs::entities::instances::instance::Instance;
use jagua_rs::fsize;
use jagua_rs::io::parser::Parser;
use jagua_rs::util::config::{CDEConfig, SPSurrogateConfig};
use jagua_rs::util::polygon_simplification::PolySimplConfig;
use ordered_float::OrderedFloat;
use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};
use std::path::Path;
use std::time::{Duration, Instant};
use gls_strip_packing::opt::post_optimizer::compress;
use gls_strip_packing::util::io::layout_to_svg::s_layout_to_svg;

const RNG_SEED: Option<usize> = None;

const GLS_TIME_LIMIT_S: u64 = 18 * 60;

const POST_TIME_LIMIT_S: u64 = 2 * 60;

fn main() {
    //the input file is the first argument
    let args: Vec<String> = std::env::args().collect();
    let n_runs_total = args[1].parse::<usize>().unwrap();
    let json_instance = io::read_json_instance(Path::new(&args[2]));

    let n_runs_per_iter = num_cpus::get_physical() / N_WORKERS;
    let n_iterations = (n_runs_total as fsize / n_runs_per_iter as fsize).ceil() as usize;

    println!(
        "Starting benchmark for {} ({}x{} runs across {} cores, {}s timelimit)",
        json_instance.name, n_iterations, n_runs_per_iter, num_cpus::get_physical(), GLS_TIME_LIMIT_S + POST_TIME_LIMIT_S
    );

    let cde_config = CDEConfig {
        quadtree_depth: 4,
        hpg_n_cells: 0,
        item_surrogate_config: SPSurrogateConfig {
            pole_coverage_goal: 0.95,
            max_poles: 20,
            n_ff_poles: 2,
            n_ff_piers: 0,
        },
    };

    let parser = Parser::new(PolySimplConfig::Disabled, cde_config, true);
    let instance = parser.parse(&json_instance);

    let sp_instance = match instance.clone() {
        Instance::SP(spi) => spi,
        _ => panic!("Expected SPInstance"),
    };

    let mut rng = match RNG_SEED {
        Some(seed) => {
            println!("Using seed: {}", seed);
            SmallRng::seed_from_u64(seed as u64)
        }
        None => {
            let seed = rand::random();
            println!("No seed provided, using: {}", seed);
            SmallRng::seed_from_u64(seed)
        }
    };

    let constr_search_config = SearchConfig {
        n_bin_samples: 1000,
        n_focussed_samples: 0,
        n_coord_descents: 3,
    };

    let mut final_solutions = vec![];

    for i in 0..n_iterations {
        println!("Starting iter {}/{}", i + 1, n_iterations);
        let mut iter_solutions = vec![None; n_runs_per_iter];
        rayon::scope(|s| {
            for (j, solution_slice) in iter_solutions.iter_mut().enumerate() {
                let thread_rng = SmallRng::seed_from_u64(rng.random());
                let svg_output_dir = format!("{}_{}_{}", SVG_OUTPUT_DIR, json_instance.name, i * n_runs_per_iter + j);
                let instance = sp_instance.clone();

                s.spawn(|_| {
                    let mut constr_builder = ConstructiveBuilder::new(
                        instance,
                        cde_config,
                        thread_rng,
                        constr_search_config,
                    );
                    constr_builder.build();

                    let instance = constr_builder.instance;
                    let problem = constr_builder.prob;
                    let rng = constr_builder.rng;

                    let mut gls_opt = GLSOrchestrator::new(problem, instance, rng, svg_output_dir);

                    let solution = gls_opt.solve(Duration::from_secs(GLS_TIME_LIMIT_S));

                    let post_solution = compress(&mut gls_opt, &solution, Instant::now().add(Duration::from_secs(POST_TIME_LIMIT_S)));

                    println!("[POST] from {:.3}% to {:.3}% ({:.3}%)", solution.usage * 100.0, post_solution.usage * 100.0, (solution.usage - post_solution.usage) * 100.0);
                    *solution_slice = Some(post_solution);
                })
            }
        });
        final_solutions.extend(iter_solutions.into_iter().flatten());
    }

    //print statistics about the solutions, print best, worst, median and average
    let (final_widths, final_usages): (Vec<fsize>, Vec<fsize>) = final_solutions
        .into_iter()
        .map(|s| {
            let width = s.layout_snapshots[0].bin.bbox().width();
            let usage = s.layout_snapshots[0].usage;
            (width, usage * 100.0)
        })
        .unzip();

    println!("final widths: {:?}", &final_widths);
    println!("final usages: {:?}", &final_usages);

    println!("----------------- WIDTH -----------------");
    println!(
        "Worst: {}",
        final_widths
            .iter()
            .max_by_key(|&x| OrderedFloat(*x))
            .unwrap()
    );
    println!("25per: {}", calculate_percentile(&final_widths, 0.75));
    println!("Med: {}", calculate_median(&final_widths));
    println!("75per: {}", calculate_percentile(&final_widths, 0.25));
    println!(
        "Best: {}",
        final_widths
            .iter()
            .min_by_key(|&x| OrderedFloat(*x))
            .unwrap()
    );
    println!("Avg: {}", calculate_average(&final_widths));
    println!("Stddev: {}", calculate_stddev(&final_widths));
    println!();
    println!("----------------- USAGE -----------------");
    println!(
        "Worst: {}",
        final_usages
            .iter()
            .min_by_key(|&x| OrderedFloat(*x))
            .unwrap()
    );
    println!("25per: {}", calculate_percentile(&final_usages, 0.25));
    println!("Median: {}", calculate_median(&final_usages));
    println!("75per: {}", calculate_percentile(&final_usages, 0.75));
    println!(
        "Best: {}",
        final_usages
            .iter()
            .max_by_key(|&x| OrderedFloat(*x))
            .unwrap()
    );
    println!("Avg: {}", calculate_average(&final_usages));
    println!("Stddev: {}", calculate_stddev(&final_usages));
}

//mimics Excel's percentile function
pub fn calculate_percentile(v: &[fsize], pct: fsize) -> fsize {
    // Validate input
    assert!(!v.is_empty(), "Cannot compute percentile of an empty slice");
    assert!(
        pct >= 0.0 && pct <= 1.0,
        "Percent must be between 0.0 and 1.0"
    );

    // Create a sorted copy of the data
    let mut sorted = v.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted.len();
    // Compute the rank using Excel's formula (1-indexed):
    // k = pct * (n - 1) + 1
    let k = pct * ((n - 1) as f64) + 1.0;

    // Determine the lower and upper indices (still 1-indexed)
    let lower_index = k.floor() as usize;
    let upper_index = k.ceil() as usize;
    let fraction = k - (lower_index as f64);

    // Convert indices to 0-indexed by subtracting 1
    let lower_value = sorted[lower_index - 1];
    let upper_value = sorted[upper_index - 1];

    // If k is an integer, fraction is 0 so this returns lower_value exactly.
    lower_value + fraction * (upper_value - lower_value)
}

pub fn calculate_median(v: &[fsize]) -> fsize {
    calculate_percentile(v, 0.5)
}

pub fn calculate_average(v: &[fsize]) -> fsize {
    v.iter().sum::<fsize>() / v.len() as fsize
}

pub fn calculate_stddev(v: &[fsize]) -> fsize {
    let avg = calculate_average(v);
    (v.iter().map(|x| (x - avg).powi(2)).sum::<fsize>() / v.len() as fsize).sqrt()
}
