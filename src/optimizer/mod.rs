use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use crate::config::*;
use crate::optimizer::lbf::LBFBuilder;
use crate::optimizer::separator::Separator;
use crate::FMT;
use jagua_rs::entities::instances::strip_packing::SPInstance;
use jagua_rs::entities::problems::problem_generic::ProblemGeneric;
use jagua_rs::entities::problems::strip_packing::strip_width;
use jagua_rs::entities::solution::Solution;
use log::{debug, info, warn};
use rand::prelude::{IteratorRandom, SmallRng};
use rand::Rng;
use rand_distr::Distribution;
use rand_distr::Normal;
use std::time::{Duration, Instant};
use jagua_rs::entities::instances::instance_generic::InstanceGeneric;
use ordered_float::OrderedFloat;

pub mod lbf;
pub mod separator;
mod separator_worker;

// All high-level heuristic logic
pub fn optimize(instance: SPInstance, rng: SmallRng, output_folder_path: String, mut terminator: Terminator, time_limit: Duration) -> Solution {
    let builder = LBFBuilder::new(instance, CDE_CONFIG, rng, LBF_SAMPLE_CONFIG).construct();
    let mut expl_separator = Separator::new(builder.instance, builder.prob, builder.rng, output_folder_path.clone(), 0, SEP_CONFIG_EXPLORE);

    terminator.set_timeout_from_now(time_limit.mul_f32(EXPLORE_TIME_RATIO));
    let solutions = explore(&mut expl_separator, &terminator);
    let mut best_sol = solutions.last().unwrap().clone();

    let mut cmpr_separator = Separator::new(expl_separator.instance, expl_separator.prob, expl_separator.rng, expl_separator.output_svg_folder, expl_separator.svg_counter, SEPARATOR_CONFIG_COMPRESS);

    for (step,time_ratio) in COMPRESS_STEPS.iter().zip(COMPRESS_TIME_RATIOS.iter()) {
        terminator.set_timeout_from_now(time_limit.mul_f32(*time_ratio)).reset_ctrlc();
        let cmpr_sol = compress(&mut cmpr_separator, &best_sol, &terminator, *step);
        best_sol = cmpr_sol;
    }

    best_sol
}

pub fn explore(sep: &mut Separator, term: &Terminator) -> Vec<Solution> {
    let mut current_width = sep.prob.occupied_width();
    let mut best_width = current_width;

    let mut feasible_solutions = vec![sep.prob.create_solution(None)];

    sep.export_svg(None, "init", false);
    info!("[EXPL] starting optimization with initial width: {:.3} ({:.3}%)",current_width,sep.prob.usage() * 100.0);

    let mut solution_pool: Vec<(Solution, f32)> = vec![];

    while !term.is_kill() {
        let local_best = sep.separate_layout(&term);
        let total_overlap = local_best.1.get_total_overlap();

        if total_overlap == 0.0 {
            //layout is successfully separated
            if current_width < best_width {
                info!("[EXPL] new best at width: {:.3} ({:.3}%)",current_width,sep.prob.usage() * 100.0);
                best_width = current_width;
                feasible_solutions.push(local_best.0.clone());
                sep.export_svg(Some(local_best.0.clone()), "f", false);
            }
            let next_width = current_width * (1.0 - EXPLORE_SHRINK_STEP);
            info!("[EXPL] shrinking width by {}%: {:.3} -> {:.3}", EXPLORE_SHRINK_STEP * 100.0, current_width, next_width);
            sep.change_strip_width(next_width, None);
            current_width = next_width;
            solution_pool.clear();
        } else {
            info!("[EXPL] layout separation unsuccessful, exporting min overlap solution");
            sep.export_svg(Some(local_best.0.clone()), "o", false);

            //layout was not successfully separated, add to local bests
            match solution_pool.binary_search_by(|(_, o)| o.partial_cmp(&total_overlap).unwrap()) {
                Ok(idx) | Err(idx) => solution_pool.insert(idx, (local_best.0.clone(), total_overlap)),
            }

            //restore to a random solution from the tabu list, better solutions have more chance to be selected
            let selected_sol = {
                //sample a value in range [0.0, 1.0[ from a normal distribution
                let distr = Normal::new(0.0, EXPLORE_SOL_DISTR_STDDEV).unwrap();
                let sample = distr.sample(&mut sep.rng).abs().min(0.999);
                //map it to the range of the solution pool
                let selected_idx = (sample * solution_pool.len() as f32) as usize;

                let (selected_sol, overlap) = &solution_pool[selected_idx];
                info!("[EXPL] selected starting solution {}/{} from solution pool (o: {})", selected_idx, solution_pool.len(), FMT.fmt2(*overlap));
                selected_sol
            };

            //restore and swap two large items
            sep.rollback(selected_sol, None);
            swap_large_pair_of_items(sep);
        }
    }

    info!("[EXPL] time limit reached, best solution found: {:.3} ({:.3}%)",best_width,feasible_solutions.last().unwrap().usage * 100.0);

    feasible_solutions
}

pub fn compress(sep: &mut Separator, init: &Solution, term: &Terminator, shrink_ratio_step: f32) -> Solution {
    info!("[CMPR] attempting to compress in steps of {}%", shrink_ratio_step * 100.0);
    let mut best = init.clone();
    while !term.is_kill() {
        match attempt_to_compress(sep, &best, shrink_ratio_step, &term) {
            Some(compacted_sol) => {
                info!("[CMPR] compressed to {:.3} ({:.3}%)", strip_width(&compacted_sol), compacted_sol.usage * 100.0);
                sep.export_svg(Some(compacted_sol.clone()), "p", false);
                best = compacted_sol;
            }
            None => {
                info!("[CMPR] compression unsuccessful");
            }
        }
    }
    info!("[CMPR] finished compression, improved from {:.3}% to {:.3}% (+{:.3}%)", init.usage * 100.0, best.usage * 100.0, (best.usage - init.usage) * 100.0);
    best
}


fn attempt_to_compress(sep: &mut Separator, init: &Solution, r_shrink: f32, term: &Terminator) -> Option<Solution> {
    //restore to the initial solution and width
    sep.change_strip_width(strip_width(&init), None);
    sep.rollback(&init, None);

    //shrink the bin at a random position
    let new_width = strip_width(init) * (1.0 - r_shrink);
    let split_pos = sep.rng.random_range(0.0..sep.prob.strip_width());
    sep.change_strip_width(new_width, Some(split_pos));

    //try to separate layout, if all overlap is eliminated, return the solution
    let (compacted_sol, ot) = sep.separate_layout(term);
    match ot.get_total_overlap() == 0.0 {
        true => Some(compacted_sol),
        false => None,
    }
}

fn swap_large_pair_of_items(sep: &mut Separator) {
    let large_area_ch_area_cutoff = sep.instance.items().iter()
        .map(|(item, _)| item.shape.surrogate().convex_hull_area)
        .max_by_key(|&x| OrderedFloat(x))
        .unwrap() * LARGE_AREA_CH_AREA_CUTOFF_RATIO;

    let layout = &sep.prob.layout;
    let (pk1, pi1) = layout.placed_items.iter()
        .filter(|(_, pi)| pi.shape.surrogate().convex_hull_area > large_area_ch_area_cutoff)
        .choose(&mut sep.rng)
        .unwrap();

    let (pk2, pi2) = layout.placed_items.iter()
        .filter(|(_, pi)| pi.item_id != pi1.item_id)
        .filter(|(_, pi)| pi.shape.surrogate().convex_hull_area > large_area_ch_area_cutoff)
        .choose(&mut sep.rng)
        .unwrap_or(layout.placed_items.iter()
            .filter(|(pk2, _)| *pk2 != pk1)
            .choose(&mut sep.rng).unwrap());

    let dt1 = pi1.d_transf;
    let dt2 = pi2.d_transf;

    info!("[EXPL] swapped two large items (ids: {} <-> {})", pi1.item_id, pi2.item_id);

    sep.move_item(pk1, dt2);
    sep.move_item(pk2, dt1);
}

#[derive(Debug, Clone)]
pub struct Terminator {
    pub timeout: Option<Instant>,
    pub ctrlc: Arc<AtomicBool>,
}

impl Terminator {
    /// Creates a dummy terminator that will never terminate
    pub fn dummy() -> Self {
        Terminator {
            timeout: None,
            ctrlc: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Only call this function once, it will set up a handler for Ctrl-C
    pub fn new_with_ctrlc_handler() -> Self {
        let ctrlc = Arc::new(AtomicBool::new(false));
        let c = ctrlc.clone();

        ctrlc::set_handler(move || {
            warn!(" terminating...");
            c.store(true, Ordering::SeqCst);
        }).expect("Error setting Ctrl-C handler");

        Terminator {
            timeout: None,
            ctrlc,
        }
    }
    pub fn is_kill(&self) -> bool {
        self.timeout.map_or(false, |timeout| Instant::now() > timeout)
            || self.ctrlc.load(Ordering::SeqCst)
    }

    pub fn reset_ctrlc(&self) -> &Self {
        self.ctrlc.store(false, Ordering::SeqCst);
        self
    }

    pub fn set_timeout_from_now(&mut self, timeout: Duration) -> &mut Self {
        self.timeout = Some(Instant::now() + timeout);
        self
    }

    pub fn clear_timeout(&mut self) -> &mut Self {
        self.timeout = None;
        self
    }
}