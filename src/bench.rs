extern crate core;
use sparrow::util::terminator::Terminator;

use ordered_float::OrderedFloat;
use rand::{Rng, RngCore, SeedableRng};
use sparrow::config::*;
use sparrow::optimizer::lbf::LBFBuilder;
use sparrow::optimizer::separator::Separator;
use sparrow::util::io;
use std::env::args;
use std::fs;
use std::path::Path;
use std::time::Duration;
use jagua_rs::Instant;

use anyhow::Result;
use jagua_rs::io::import::Importer;
use jagua_rs::io::svg::s_layout_to_svg;
use rand_xoshiro::Xoshiro256PlusPlus;
use sparrow::consts::{DEFAULT_COMPRESS_TIME_RATIO, DEFAULT_EXPLORE_TIME_RATIO, DRAW_OPTIONS, LBF_SAMPLE_CONFIG};
use sparrow::optimizer::compress::compression_phase;
use sparrow::optimizer::explore::exploration_phase;
use sparrow::util::listener::DummySolListener;
use sparrow::util::terminator::BasicTerminator;

pub const OUTPUT_DIR: &str = "output";

fn main() -> Result<()> {
    let mut config = DEFAULT_SPARROW_CONFIG;

    //the input file is the first argument
    let input_file_path = args().nth(1).expect("first argument must be the input file");
    let time_limit: Duration = args().nth(2).expect("second argument must be the time limit [s]")
        .parse::<u64>().map(|s| Duration::from_secs(s))
        .expect("second argument must be the time limit [s]");
    let n_runs_total = args().nth(3).expect("third argument must be the number of runs")
        .parse().expect("third argument must be the number of runs");

    fs::create_dir_all(OUTPUT_DIR).expect("could not create output directory");

    println!("[BENCH] git commit hash: {}", get_git_commit_hash());
    println!("[BENCH] system time: {}", jiff::Timestamp::now());

    let mut rng = match config.rng_seed {
        Some(seed) => {
            println!("[BENCH] using provided seed: {}", seed);
            Xoshiro256PlusPlus::seed_from_u64(seed as u64)
        }
        None => {
            let seed = rand::random();
            println!("[BENCH] no seed provided, using: {}", seed);
            Xoshiro256PlusPlus::seed_from_u64(seed)
        }
    };

    config.expl_cfg.time_limit = time_limit.mul_f32(DEFAULT_EXPLORE_TIME_RATIO);
    config.cmpr_cfg.time_limit = time_limit.mul_f32(DEFAULT_COMPRESS_TIME_RATIO);

    let n_runs_per_iter = (num_cpus::get_physical() / config.expl_cfg.separator_config.n_workers).min(n_runs_total);
    let n_batches = (n_runs_total as f32 / n_runs_per_iter as f32).ceil() as usize;

    let ext_instance = io::read_spp_instance_json(Path::new(&input_file_path))?;

    println!(
        "[BENCH] starting bench for {} ({}x{} runs across {} cores, {:?} timelimit)",
        ext_instance.name, n_batches, n_runs_per_iter, num_cpus::get_physical(), time_limit
    );

    let importer = Importer::new(config.cde_config, config.poly_simpl_tolerance, config.min_item_separation, config.narrow_concavity_cutoff_ratio);
    let instance = jagua_rs::probs::spp::io::import(&importer, &ext_instance)?;

    let mut final_solutions = vec![];


    for i in 0..n_batches {
        println!("[BENCH] batch {}/{}", i + 1, n_batches);
        println!("[BENCH] system time: {}", jiff::Timestamp::now());
        let mut iter_solutions = vec![None; n_runs_per_iter];
        rayon::scope(|s| {
            for (j, sol_slice) in iter_solutions.iter_mut().enumerate() {
                let bench_idx = i * n_runs_per_iter + j;
                let instance = instance.clone();
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(rng.random());
                let mut terminator = BasicTerminator::new();

                s.spawn(move |_| {
                    let mut next_rng = || Xoshiro256PlusPlus::seed_from_u64(rng.next_u64());
                    let builder = LBFBuilder::new(instance.clone(), next_rng(), LBF_SAMPLE_CONFIG).construct();
                    let mut expl_separator = Separator::new(builder.instance, builder.prob, next_rng(), config.expl_cfg.separator_config);

                    terminator.new_timeout(config.expl_cfg.time_limit);
                    let solutions = exploration_phase(&instance, &mut expl_separator, &mut DummySolListener, &terminator, &config.expl_cfg);
                    let final_explore_sol = solutions.last().expect("no solutions found during exploration");

                    let start_comp = Instant::now();

                    terminator.new_timeout(config.cmpr_cfg.time_limit);
                    let mut cmpr_separator = Separator::new(expl_separator.instance, expl_separator.prob, next_rng(), config.cmpr_cfg.separator_config);
                    let cmpr_sol = compression_phase(&instance, &mut cmpr_separator, final_explore_sol, &mut DummySolListener, &terminator, &config.cmpr_cfg);

                    println!("[BENCH] [id:{:>3}] finished, expl: {:.3}% ({}s), cmpr: {:.3}% (+{:.3}%) ({}s)",
                             bench_idx,
                             final_explore_sol.density(&instance) * 100.0, time_limit.mul_f32(DEFAULT_EXPLORE_TIME_RATIO).as_secs(),
                             cmpr_sol.density(&instance) * 100.0,
                             cmpr_sol.density(&instance) * 100.0 - final_explore_sol.density(&instance) * 100.0,
                             start_comp.elapsed().as_secs()
                    );

                    io::write_svg(
                        &s_layout_to_svg(&cmpr_sol.layout_snapshot, &instance, DRAW_OPTIONS, &*format!("final_bench_{}", bench_idx)),
                        Path::new(&format!("{OUTPUT_DIR}/final_bench_{}.svg", bench_idx)),
                        log::Level::Info,
                    ).expect(&*format!("could not write svg output of bench {}", bench_idx));

                    *sol_slice = Some(cmpr_sol);
                })
            }
        });
        final_solutions.extend(iter_solutions.into_iter().flatten());
    }

    //print statistics about the solutions, print best, worst, median and average
    let (final_widths, final_usages): (Vec<f32>, Vec<f32>) = final_solutions
        .iter()
        .map(|s| {
            let width = s.strip_width();
            let usage = s.layout_snapshot.density(&instance);
            (width, usage * 100.0)
        })
        .unzip();

    let best_final_solution = final_solutions.iter().max_by_key(|s| OrderedFloat(s.density(&instance))).unwrap();

    io::write_svg(
        &s_layout_to_svg(&best_final_solution.layout_snapshot, &instance, DRAW_OPTIONS, "final_best"),
        Path::new(format!("{OUTPUT_DIR}/final_best_{}.svg", ext_instance.name).as_str()),
        log::Level::Info,
    )?;

    println!("==== BENCH FINISHED ====");

    println!("widths:\n{:?}", &final_widths);
    println!("usages:\n{:?}", &final_usages);

    println!("---- WIDTH STATS ----");
    println!("worst:  {:.3}", final_widths.iter().max_by_key(|&x| OrderedFloat(*x)).unwrap());
    println!("25%:    {:.3}", calculate_percentile(&final_widths, 0.75));
    println!("med:    {:.3}", calculate_median(&final_widths));
    println!("75%:    {:.3}", calculate_percentile(&final_widths, 0.25));
    println!("best:   {:.3}", final_widths.iter().min_by_key(|&x| OrderedFloat(*x)).unwrap());
    println!("avg:    {:.3}", calculate_average(&final_widths));
    println!("stddev: {:.3}", calculate_stddev(&final_widths));
    println!("---- USAGE STATS ----");
    println!("worst:  {:.3}", final_usages.iter().min_by_key(|&x| OrderedFloat(*x)).unwrap());
    println!("25%:    {:.3}", calculate_percentile(&final_usages, 0.25));
    println!("median: {:.3}", calculate_median(&final_usages));
    println!("75%:    {:.3}", calculate_percentile(&final_usages, 0.75));
    println!("best:   {:.3}", final_usages.iter().max_by_key(|&x| OrderedFloat(*x)).unwrap());
    println!("avg:    {:.3}", calculate_average(&final_usages));
    println!("stddev: {:.3}", calculate_stddev(&final_usages));
    println!("======================");
    println!("[BENCH] system time: {}", jiff::Timestamp::now());

    Ok(())
}


pub fn calculate_percentile(v: &[f32], pct: f32) -> f32 {
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
    let k = pct * (n - 1) as f32 + 1.0;

    // Determine the lower and upper indices (still 1-indexed)
    let lower_index = k.floor() as usize;
    let upper_index = k.ceil() as usize;
    let fraction = k - (lower_index as f32);

    // Convert indices to 0-indexed by subtracting 1
    let lower_value = sorted[lower_index - 1];
    let upper_value = sorted[upper_index - 1];

    // If k is an integer, fraction is 0 so this returns lower_value exactly.
    lower_value + fraction * (upper_value - lower_value)
}

pub fn calculate_median(v: &[f32]) -> f32 {
    calculate_percentile(v, 0.5)
}

pub fn calculate_average(v: &[f32]) -> f32 {
    v.iter().sum::<f32>() / v.len() as f32
}

pub fn calculate_stddev(v: &[f32]) -> f32 {
    let avg = calculate_average(v);
    (v.iter().map(|x| (x - avg).powi(2)).sum::<f32>() / v.len() as f32).sqrt()
}

pub fn get_git_commit_hash() -> String {
    let output = std::process::Command::new("git")
        .args(&["rev-parse", "HEAD"])
        .output()
        .expect("Failed to execute git command");

    match output.status.success() {
        true => String::from_utf8_lossy(&output.stdout).trim().to_string(),
        false => "unknown".to_string(),
    }
}
