use crate::io::layout_to_svg::{layout_to_svg, s_layout_to_svg};
use crate::io::svg_util::SvgDrawOptions;
use crate::opt::tabu::TabuList;
use crate::overlap::overlap_tracker;
use crate::overlap::overlap_tracker::{OTSnapshot, OverlapTracker};
use crate::sample::eval::overlapping_evaluator::OverlappingSampleEvaluator;
use crate::sample::eval::SampleEval;
use crate::sample::search;
use crate::sample::search::SearchConfig;
use crate::{io, FMT, SVG_OUTPUT_DIR};
use float_cmp::approx_eq;
use itertools::{sorted, Itertools};
use jagua_rs::entities::bin::Bin;
use jagua_rs::entities::instances::instance_generic::InstanceGeneric;
use jagua_rs::entities::instances::strip_packing::SPInstance;
use jagua_rs::entities::layout::Layout;
use jagua_rs::entities::placed_item::PItemKey;
use jagua_rs::entities::placing_option::PlacingOption;
use jagua_rs::entities::problems::problem_generic::{ProblemGeneric, STRIP_LAYOUT_IDX};
use jagua_rs::entities::problems::strip_packing::SPProblem;
use jagua_rs::entities::solution::Solution;
use jagua_rs::fsize;
use jagua_rs::geometry::d_transformation::DTransformation;
use jagua_rs::geometry::geo_enums::GeoRelation;
use jagua_rs::geometry::geo_traits::{Shape, Transformable};
use jagua_rs::geometry::primitives::aa_rectangle::AARectangle;
use jagua_rs::util::fpa::FPA;
use log::{debug, info, warn};
use ordered_float::OrderedFloat;
use rand::prelude::IteratorRandom;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::Distribution;
use rand_distr::Normal;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::{split, ParallelIterator};
use std::char::decode_utf16;
use std::cmp::Reverse;
use std::collections::VecDeque;
use std::iter;
use std::ops::Range;
use std::path::Path;
use std::time::{Duration, Instant};
use tap::Tap;
use crate::opt::gls_worker::GLSWorker;

pub const N_ITER_NO_IMPROVEMENT: usize = 50;

pub const N_STRIKES: usize = 5;

//TODO: Let the shrink rate depend on the time left. example: 0.5% for the first 3/4, 0.05% for the following 3/4, 0.005% for the last 1/16
pub const R_SHRINK: fsize = 0.005;
//const R_EXPAND: fsize = 0.003;

pub const N_UNIFORM_SAMPLES: usize = 100;
pub const N_COORD_DESCENTS: usize = 2;
pub const TABU_SIZE: usize = 10_000;
pub const JUMP_COOLDOWN: usize = 5;

pub const N_THREADS: usize = 2;

pub const OT_MAX_INCREASE : fsize = 2.0;
pub const OT_MIN_INCREASE: fsize = 1.2;
pub const OT_DECAY: fsize = 0.95;

pub struct GLSOrchestrator {
    pub instance: SPInstance,
    pub rng: SmallRng,
    pub master_prob: SPProblem,
    pub master_ot: OverlapTracker,
    pub workers: Vec<GLSWorker>,
    pub output_folder: String,
    pub svg_counter: usize,
    pub tabu_list: TabuList,
}

impl GLSOrchestrator {
    pub fn new(
        problem: SPProblem,
        instance: SPInstance,
        mut rng: SmallRng,
        output_folder: String,
    ) -> Self {
        let overlap_tracker = OverlapTracker::from_layout(&problem.layout);
        let tabu_list = TabuList::new(TABU_SIZE, &instance);
        let workers = (0..N_THREADS)
            .map(|_| GLSWorker{
                instance: instance.clone(),
                prob: problem.clone(),
                ot: overlap_tracker.clone(),
                rng: SmallRng::seed_from_u64(rng.random()),
                ch_area_cutoff: tabu_list.ch_area_cutoff,
            }).collect();
        Self {
            master_prob: problem.clone(),
            instance,
            rng,
            master_ot: overlap_tracker.clone(),
            workers,
            svg_counter: 0,
            output_folder,
            tabu_list,
        }
    }

    pub fn solve(&mut self, time_out: Duration) -> Solution {
        let mut current_width = self.master_prob.occupied_width();
        let (mut best_feasible_solution, mut best_width) = (self.master_prob.create_solution(None), current_width);

        self.write_to_disk(None, true);
        info!("[GLS] starting optimization with initial width: {:.3} ({:.3}%)", current_width, self.master_prob.usage() * 100.0);

        let start = Instant::now();

        while start.elapsed() < time_out {
            let local_best = self.separate_layout();
            let total_overlap = self.master_ot.get_total_overlap();

            if total_overlap == 0.0 {
                //layout is successfully separated
                if current_width < best_width {
                    info!("[GLS] new best width at : {:.3} ({:.3}%)", current_width, self.master_prob.usage() * 100.0);
                    best_width = current_width;
                    best_feasible_solution = local_best.0.clone();
                    self.write_to_disk(Some(best_feasible_solution.clone()), true);
                }
                let next_width = current_width * (1.0 - R_SHRINK);
                info!("[GLS] shrinking width from {:.3} to {:.3}", current_width, next_width);
                self.change_strip_width(next_width, None);
                current_width = next_width;
            }
            else {
                //layout was not successfully separated
                self.tabu_list.push(local_best.0.clone(), total_overlap);
                info!("[GLS] layout separation unsuccessful, adding local best to tabu list");

                //restore to a random solution from the tabu list, better solutions have more chance to be selected
                let selected_sol = {
                    let sorted_sols = self.tabu_list.list.iter()
                        .filter(|(sol, eval)| sol.layout_snapshots[0].bin.bbox().width() == current_width)
                        .sorted_by_key(|(_, eval)| OrderedFloat(*eval))
                        .collect_vec();

                    let mut distr = Normal::new(0.0_f64, sorted_sols.len() as fsize / 5.0).unwrap();
                    let selected_idx = (distr.sample(&mut self.rng).abs().floor() as usize).min(sorted_sols.len() - 1);
                    let selected = sorted_sols.get(selected_idx).unwrap();
                    selected.0.clone()
                };

                self.rollback(&selected_sol, None);
            }
        }

        info!("[GLS] time limit reached, returning best solution: {:.3} ({:.3}%)", best_width, best_feasible_solution.layout_snapshots[0].usage * 100.0);
        self.write_to_disk(Some(best_feasible_solution.clone()), true);

        best_feasible_solution
    }

    pub fn separate_layout(&mut self) -> (Solution, OTSnapshot, usize) {
        let mut min_overlap = fsize::INFINITY;
        let mut min_overlap_sol: Option<(Solution, OTSnapshot)> = None;

        let mut n_strikes = 0;
        let mut n_iter = 0;
        let mut n_items_moved = 0;
        let start = Instant::now();

        while n_strikes < N_STRIKES {
            let mut n_iter_no_improvement = 0;

            if let Some(min_overlap_solution) = min_overlap_sol.as_ref() {
                info!("[s:{n_strikes}] Rolling back to min overlap");
                self.rollback(&min_overlap_solution.0, Some(&min_overlap_solution.1));
            }

            let initial_strike_overlap = self.master_ot.get_total_overlap();
            info!("[s:{n_strikes}] initial overlap: {}", FMT.fmt2(initial_strike_overlap));

            while n_iter_no_improvement < N_ITER_NO_IMPROVEMENT {
                let (overlap_before, w_overlap_before) = (self.master_ot.get_total_overlap(), self.master_ot.get_total_weighted_overlap());
                let n_moves = self.modify();
                let (overlap, w_overlap) = (self.master_ot.get_total_overlap(), self.master_ot.get_total_weighted_overlap());

                debug!("[s:{n_strikes}, i:{n_iter}]    w_o: {} -> {}, o: {} -> {}, n_mov: {}, (min o: {})", FMT.fmt2(w_overlap_before), FMT.fmt2(w_overlap), FMT.fmt2(overlap_before), FMT.fmt2(overlap), n_moves, FMT.fmt2(min_overlap));
                debug_assert!(FPA(w_overlap) <= FPA(w_overlap_before), "weighted overlap increased: {} -> {}", FMT.fmt2(w_overlap_before), FMT.fmt2(w_overlap));

                if overlap == 0.0 {
                    //layout is successfully separated
                    info!("[s:{n_strikes}, i:{n_iter}] (S) w_o: {} -> {}, o: {} -> {}, n_mov: {}, (min o: {})", FMT.fmt2(w_overlap_before), FMT.fmt2(w_overlap), FMT.fmt2(overlap_before), FMT.fmt2(overlap), n_moves, FMT.fmt2(min_overlap));
                    return (self.master_prob.create_solution(None), self.master_ot.create_snapshot(), n_items_moved);
                } else if overlap < min_overlap {
                    //layout is not separated, but absolute overlap is better than before

                    let sol = self.master_prob.create_solution(None);

                    if !self.tabu_list.sol_is_tabu(&sol) {
                        info!("[s:{n_strikes}, i:{n_iter}] (*) w_o: {} -> {}, o: {} -> {}, n_mov: {}, (min o: {})", FMT.fmt2(w_overlap_before), FMT.fmt2(w_overlap), FMT.fmt2(overlap_before), FMT.fmt2(overlap), n_moves, FMT.fmt2(min_overlap));
                        min_overlap = overlap;
                        min_overlap_sol = Some((sol, self.master_ot.create_snapshot()));
                        n_iter_no_improvement = 0;
                    } else {
                        info!("[s:{n_strikes}, i: {n_iter}] tabu solution encountered, swapping two items");
                        self.swap_tabu_item();
                    }
                } else {
                    n_iter_no_improvement += 1;
                }

                self.master_ot.increment_weights();
                n_items_moved += n_moves;
                n_iter += 1;
            }
            info!("[s:{n_strikes}, i: {n_iter}] {} iter no improvement, min overlap: {}", n_iter_no_improvement, FMT.fmt2(min_overlap));
            if initial_strike_overlap * 0.98 <= min_overlap {
                info!("[s:{n_strikes}, i: {n_iter}] no substantial improvement, adding strike");
                n_strikes += 1;
            } else {
                info!("[s:{n_strikes}, i: {n_iter}] improvement, resetting strikes");
                n_strikes = 0;
            }
            self.write_to_disk(None, true);
        }
        info!("[GLS] strike limit reached ({}), moves/s: {}, iter/s: {}, time: {}ms",
            n_strikes,
            (n_items_moved as fsize / start.elapsed().as_secs_f64()) as usize,
            (n_iter as fsize / start.elapsed().as_secs_f64()) as usize,
            start.elapsed().as_millis());

        let min_overlap_sol = min_overlap_sol.expect("no solution found");

        (min_overlap_sol.0, min_overlap_sol.1, n_items_moved)
    }

    pub fn modify(&mut self) -> usize {
        let master_sol = self.master_prob.create_solution(None);

        let n_movements = self.workers.par_iter_mut()
            .map(|worker| {
                // Sync the workers with the master
                worker.load(&master_sol, &self.master_ot);
                // Let them modify
                let n_moves = worker.separate();
                n_moves
            })
            .sum();

        debug!("optimizers w_o's: {:?}", self.workers.iter().map(|opt| opt.ot.get_total_weighted_overlap()).collect_vec());

        // Check which worker has the lowest total weighted overlap
        let best_opt = self.workers.iter_mut()
            .min_by_key(|opt| OrderedFloat(opt.ot.get_total_weighted_overlap()))
            .map(|opt| (opt.prob.create_solution(None), &opt.ot))
            .unwrap();

        // Sync the master with the best optimizer
        self.master_prob.restore_to_solution(&best_opt.0);
        self.master_ot = best_opt.1.clone();

        n_movements
    }

    pub fn rollback(&mut self, solution: &Solution, ots: Option<&OTSnapshot>) {
        self.master_prob.restore_to_solution(solution);

        match ots {
            Some(ots) => {
                //if an snapshot of the overlap tracker was provided, restore it
                self.master_ot.restore(ots, &self.master_prob.layout);
            }
            None => {
                //otherwise, rebuild it
                self.master_ot = OverlapTracker::from_layout(&self.master_prob.layout);
            }
        }
    }

    pub fn swap_tabu_item(&mut self) {
        info!("swapping tabu item");
        let layout = &self.master_prob.layout;
        let (pk1, pi1) = layout.placed_items.iter()
            .filter(|(_, pi)| pi.shape.surrogate().convex_hull_area > self.tabu_list.ch_area_cutoff)
            .choose(&mut self.rng)
            .unwrap();

        let (pk2, pi2) = layout.placed_items.iter()
            .filter(|(_, pi)| pi.item_id != pi1.item_id)
            .filter(|(_, pi)| pi.shape.surrogate().convex_hull_area > self.tabu_list.ch_area_cutoff)
            .choose(&mut self.rng)
            .unwrap();

        let dtransf1 = pi1.d_transf;
        let dtransf2 = pi2.d_transf;

        let new_pk1 = self.move_item(pk1, dtransf2, None);
        let new_pk2 = self.move_item(pk2, dtransf1, None);
    }

    fn move_item(&mut self, pik: PItemKey, d_transf: DTransformation, eval: Option<SampleEval>) -> PItemKey {
        debug_assert!(overlap_tracker::tracker_matches_layout(&self.master_ot, &self.master_prob.layout));

        let old_overlap = self.master_ot.get_overlap(pik);
        let old_weighted_overlap = self.master_ot.get_weighted_overlap(pik);
        let old_bbox = self.master_prob.layout.placed_items()[pik].shape.bbox();

        //Remove the item from the problem
        let old_p_opt = self.master_prob.remove_item(STRIP_LAYOUT_IDX, pik, true);
        let item = self.instance.item(old_p_opt.item_id);

        //Compute the colliding entities after the move
        let colliding_entities = {
            let shape = item.shape.clone().as_ref().clone()
                .tap_mut(|s| { s.transform(&d_transf.compose()); });

            let mut colliding_entities = vec![];
            self.master_prob.layout.cde().collect_poly_collisions(&shape, &[], &mut colliding_entities);
            colliding_entities
        };

        assert!(colliding_entities.is_empty() || !matches!(eval, Some(SampleEval::Valid(_))), "colliding entities detected for valid placement");

        let new_pk = {
            let new_p_opt = PlacingOption {
                d_transf,
                ..old_p_opt
            };

            let (_, new_pik) = self.master_prob.place_item(new_p_opt);
            new_pik
        };

        //info!("moving item {} from {:?} to {:?} ({:?}->{:?})", item.id, old_p_opt.d_transf, new_p_opt.d_transf, pik, new_pik);

        self.master_ot.register_item_move(&self.master_prob.layout, pik, new_pk);

        let new_overlap = self.master_ot.get_overlap(new_pk);
        let new_weighted_overlap = self.master_ot.get_weighted_overlap(new_pk);
        let new_bbox = self.master_prob.layout.placed_items()[new_pk].shape.bbox();

        let jumped = old_bbox.relation_to(&new_bbox) == GeoRelation::Disjoint;
        let item_big_enough = item.shape.surrogate().convex_hull_area > self.tabu_list.ch_area_cutoff;
        if jumped && item_big_enough {
            self.master_ot.register_jump(new_pk);
        }

        debug!("Moved item {} from from o: {}, wo: {} to o+1: {}, w_o+1: {} (jump: {})", item.id, FMT.fmt2(old_overlap), FMT.fmt2(old_weighted_overlap), FMT.fmt2(new_overlap), FMT.fmt2(new_weighted_overlap), jumped);

        debug_assert!(overlap_tracker::tracker_matches_layout(&self.master_ot, &self.master_prob.layout));

        new_pk
    }

    pub fn change_strip_width(&mut self, new_width: fsize, split_position: Option<fsize>) {
        let current_width = self.master_prob.strip_width();
        let delta = new_width - current_width;
        //shift all items right of the center of the strip

        let split_position = split_position.unwrap_or(current_width / 2.0);

        let shift_transf = DTransformation::new(0.0, (delta + FPA::tolerance(), 0.0));
        let items_to_shift = self.master_prob.layout.placed_items().iter()
            .filter(|(_, pi)| pi.shape.centroid().0 > split_position)
            .map(|(k, pi)| (k, pi.d_transf))
            .collect_vec();

        for (pik, dtransf) in items_to_shift {
            let new_transf = dtransf.compose().translate(shift_transf.translation());
            self.move_item(pik, new_transf.decompose(), None);
        }

        let new_bin = Bin::from_strip(
            AARectangle::new(0.0, 0.0, new_width, self.master_prob.strip_height()),
            self.master_prob.layout.bin.base_cde.config().clone(),
        );
        self.master_prob.layout.change_bin(new_bin);
        self.master_ot = OverlapTracker::from_layout(&self.master_prob.layout);

        self.workers.iter_mut().for_each(|opt| {
            *opt = GLSWorker{
                instance: self.instance.clone(),
                prob: self.master_prob.clone(),
                ot: self.master_ot.clone(),
                rng: SmallRng::seed_from_u64(self.rng.random()),
                ch_area_cutoff: self.tabu_list.ch_area_cutoff,
            };
        });

        info!("changed strip width to {}", new_width);
    }

    pub fn write_to_disk(&mut self, solution: Option<Solution>, force: bool) {
        //make sure we are in debug mode or force is true
        if !force && !cfg!(debug_assertions) {
            return;
        }

        if self.svg_counter == 0 {
            //remove all .svg files from the output folder
            let _ = std::fs::remove_dir_all(&self.output_folder);
            std::fs::create_dir_all(&self.output_folder).unwrap();
        }


        match solution {
            Some(sol) => {
                let filename = format!("{}/{}_{:.2}_s.svg", &self.output_folder, self.svg_counter, sol.layout_snapshots[0].bin.bbox().x_max);
                io::write_svg(
                    &s_layout_to_svg(&sol.layout_snapshots[0], &self.instance, SvgDrawOptions::default()),
                    Path::new(&filename),
                );
            }
            None => {
                let filename = format!("{}/{}_{:.2}.svg", &self.output_folder, self.svg_counter, self.master_prob.layout.bin.bbox().x_max);
                io::write_svg(
                    &layout_to_svg(&self.master_prob.layout, &self.instance, SvgDrawOptions::default()),
                    Path::new(&filename),
                );
            }
        }

        self.svg_counter += 1;
    }
}