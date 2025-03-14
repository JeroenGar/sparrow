use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;
use itertools::Itertools;
use jagua_rs::entities::instances::strip_packing::SPInstance;
use jagua_rs::entities::solution::Solution;
use jagua_rs::io::parser::Parser;
use jagua_rs::util::polygon_simplification::PolySimplConfig;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use sparrow::config::{CDE_CONFIG, LBF_SAMPLE_CONFIG, OUTPUT_DIR, SEPARATOR_CONFIG_COMPRESS, SEP_CONFIG_EXPLORE};
use sparrow::EXPORT_ONLY_FINAL_SVG;
use sparrow::optimizer::lbf::LBFBuilder;
use sparrow::optimizer::separator::Separator;
use sparrow::optimizer::{compress, explore, Terminator};
use sparrow::util::io;

const INSTANCES: [&str; 3] = ["swim", "trousers", "shirts"];
const N_RUNS_PER_INSTANCE: usize = 8;
const EXPL_TIME_LIMIT: Duration = Duration::from_secs(600);

const SEED: u64 = 0;

pub fn main(){
    assert!(EXPORT_ONLY_FINAL_SVG, "EXPORT_ONLY_FINAL_SVG must be true");
    // Run swim, trousers and shirts each 16 times for 10 minutes.
    // return the geomean

    let mut rng = SmallRng::seed_from_u64(SEED);

    let parser = Parser::new(PolySimplConfig::Disabled, CDE_CONFIG, true);
    let json_instances = INSTANCES.map(|name| {
        let path = format!("libs/jagua-rs/assets/{name}.json");
        parser.parse(&io::read_json_instance(Path::new(&path)))
    });
    let instances = json_instances.map(|instance| {
        match instance {
            jagua_rs::entities::instances::instance::Instance::SP(spi) => spi,
            _ => panic!("expected strip packing instance"),
        }
    });

    let final_solutions = instances.iter()
        .map(|instance| {
            bench(instance.clone(), &mut rng, N_RUNS_PER_INSTANCE, EXPL_TIME_LIMIT)
        })
        .flatten()
        .collect_vec();

    //calculate geomean usage of all solutions
    let usage_product = final_solutions.iter().map(|s| s.usage).product::<f32>();
    let geomean = usage_product.powf(1.0 / final_solutions.len() as f32);

    println!("{geomean}");
}

fn bench(instance: SPInstance,rng: &mut SmallRng, n_runs_total: usize, explore_time_limit: Duration) -> Vec<Solution> {
    let n_runs_per_iter = (num_cpus::get_physical() / SEP_CONFIG_EXPLORE.n_workers).min(n_runs_total);
    let n_batches = (n_runs_total as f32 / n_runs_per_iter as f32).ceil() as usize;

    let mut final_solutions = vec![];

    let dummy_terminator = Terminator {
        timeout: None,
        ctrlc: Arc::new(AtomicBool::new(false)),
    };

    for i in 0..n_batches {
        //println!("[BENCH] batch {}/{}", i + 1, n_batches);
        let mut iter_solutions = vec![None; n_runs_per_iter];
        rayon::scope(|s| {
            for (j, sol_slice) in iter_solutions.iter_mut().enumerate() {
                let output_folder_path = format!("{OUTPUT_DIR}/hyper_opt");
                let instance = instance.clone();
                let rng = SmallRng::seed_from_u64(rng.random());
                let mut terminator = dummy_terminator.clone();

                s.spawn(move |_| {
                    let builder = LBFBuilder::new(instance.clone(), CDE_CONFIG, rng, LBF_SAMPLE_CONFIG).construct();
                    let mut expl_separator = Separator::new(builder.instance, builder.prob, builder.rng, output_folder_path, 0, SEP_CONFIG_EXPLORE);

                    terminator.set_timeout(Some(explore_time_limit));
                    let solutions = explore(&mut expl_separator, &terminator);
                    let final_explore_sol = solutions.last().expect("no solutions found during exploration");

                    terminator.set_timeout(None);
                    terminator.reset_ctrlc();
                    let mut cmpr_separator = Separator::new(expl_separator.instance, expl_separator.prob, expl_separator.rng, expl_separator.output_svg_folder, expl_separator.svg_counter, SEPARATOR_CONFIG_COMPRESS);
                    let final_sol = compress(&mut cmpr_separator, final_explore_sol, &terminator);
                    *sol_slice = Some(final_sol);
                })
            }
        });
        final_solutions.extend(iter_solutions.into_iter().flatten());
    }
    final_solutions
}