use crate::config::{DRAW_OPTIONS, LIVE_DIR};
use crate::optimizer::separator_worker::SeparatorWorker;
use crate::overlap::tracker::{OTSnapshot, OverlapTracker};
use crate::sample::search::SampleConfig;
use crate::util::assertions::tracker_matches_layout;
use crate::util::io;
use crate::util::io::layout_to_svg::{layout_to_svg, s_layout_to_svg};
use crate::{EXPORT_LIVE_SVG, EXPORT_ONLY_FINAL_SVG, FMT};
use itertools::Itertools;
use jagua_rs::entities::bin::Bin;
use jagua_rs::entities::instances::strip_packing::SPInstance;
use jagua_rs::entities::placed_item::PItemKey;
use jagua_rs::entities::placing_option::PlacingOption;
use jagua_rs::entities::problems::problem_generic::{ProblemGeneric, STRIP_LAYOUT_IDX};
use jagua_rs::entities::problems::strip_packing::{strip_width, SPProblem};
use jagua_rs::entities::solution::Solution;
use jagua_rs::geometry::d_transformation::DTransformation;
use jagua_rs::geometry::geo_traits::Shape;
use jagua_rs::geometry::primitives::aa_rectangle::AARectangle;
use jagua_rs::util::fpa::FPA;
use log::{debug, log, Level};
use ordered_float::OrderedFloat;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use std::path::Path;
use std::time::Instant;
use crate::optimizer::Terminator;

pub struct SeparatorConfig {
    pub iter_no_imprv_limit: &'static [usize],
    pub strike_limit: usize,
    pub n_workers: usize,
    pub log_level: Level,
    pub sample_config: SampleConfig
}

pub struct Separator {
    pub instance: SPInstance,
    pub rng: SmallRng,
    pub prob: SPProblem,
    pub ot: OverlapTracker,
    pub workers: Vec<SeparatorWorker>,
    pub svg_counter: usize,
    pub output_svg_folder: String,
    pub config: SeparatorConfig
}

impl Separator {
    pub fn new(instance: SPInstance, prob: SPProblem, mut rng: SmallRng, output_svg_folder: String, svg_counter: usize, config: SeparatorConfig) -> Self {
        let overlap_tracker = OverlapTracker::new(&prob.layout);
        let workers = (0..config.n_workers).map(|_|
            SeparatorWorker {
                instance: instance.clone(),
                prob: prob.clone(),
                ot: overlap_tracker.clone(),
                rng: SmallRng::seed_from_u64(rng.random()),
                sample_config: config.sample_config.clone(),
            }).collect();

        Self {
            prob,
            instance,
            rng,
            ot: overlap_tracker,
            workers,
            svg_counter,
            output_svg_folder,
            config
        }
    }

    pub fn separate_layout(&mut self, term: &Terminator) -> (Solution, OTSnapshot) {
        let mut min_overlap_sol: (Solution, OTSnapshot) = (self.prob.create_solution(None), self.ot.create_snapshot());
        let mut min_overlap = self.ot.get_total_overlap();
        log!(self.config.log_level,"[SEP] separating at width: {:.3} and overlap: {} ", self.prob.strip_width(), FMT.fmt2(min_overlap));

        let mut n_strikes = 0;
        let mut n_iter = 0;
        let mut n_evals = 0;
        let start = Instant::now();

        while n_strikes < self.config.strike_limit && !term.is_kill() {
            let mut n_iter_no_improvement = 0;

            let initial_strike_overlap = self.ot.get_total_overlap();
            debug!("[SEP] [s:{n_strikes},i:{n_iter}]     init_o: {}",FMT.fmt2(initial_strike_overlap));

            while n_iter_no_improvement < self.config.iter_no_imprv_limit[n_strikes] {
                let (overlap_before, w_overlap_before) = (
                    self.ot.get_total_overlap(),
                    self.ot.get_total_weighted_overlap(),
                );
                n_evals += self.modify();
                let (overlap, w_overlap) = (
                    self.ot.get_total_overlap(),
                    self.ot.get_total_weighted_overlap(),
                );

                debug!("[SEP] [s:{n_strikes},i:{n_iter}] ( ) o: {} -> {}, w_o: {} -> {}, (min o: {})",FMT.fmt2(overlap_before),FMT.fmt2(overlap),FMT.fmt2(w_overlap_before),FMT.fmt2(w_overlap),FMT.fmt2(min_overlap));
                debug_assert!(FPA(w_overlap) <= FPA(w_overlap_before), "weighted overlap increased: {} -> {}", FMT.fmt2(w_overlap_before), FMT.fmt2(w_overlap));

                if overlap == 0.0 {
                    //layout is successfully separated
                    log!(self.config.log_level,"[SEP] [s:{n_strikes},i:{n_iter}] (S)  min_o: {}",FMT.fmt2(overlap));
                    return (self.prob.create_solution(None), self.ot.create_snapshot());
                } else if overlap < min_overlap {
                    //layout is not separated, but absolute overlap is better than before
                    let sol = self.prob.create_solution(None);

                    log!(self.config.log_level,"[SEP] [s:{n_strikes},i:{n_iter}] (*) min_o: {}",FMT.fmt2(overlap));
                    self.export_svg(None, "i", true);
                    if overlap < min_overlap * 0.98 {
                        //only reset the iter_no_improvement counter if the overlap improved significantly
                        n_iter_no_improvement = 0;
                    }
                    min_overlap_sol = (sol, self.ot.create_snapshot());
                    min_overlap = overlap;
                } else {
                    n_iter_no_improvement += 1;
                }

                self.ot.increment_weights();
                n_iter += 1;
            }

            if initial_strike_overlap * 0.98 <= min_overlap {
                n_strikes += 1;
            } else {
                n_strikes = 0;
            }
            self.rollback(&min_overlap_sol.0, Some(&min_overlap_sol.1));
        }
        if !term.is_kill() {
            log!(self.config.log_level,"[SEP] ended due to strike limit ({}), evals/s: {}, iter/s: {}, took {:.3}s",n_strikes,FMT.fmt2(n_evals as f64 / start.elapsed().as_secs_f64()),FMT.fmt2(n_iter as f64 / start.elapsed().as_secs_f64()),start.elapsed().as_secs());
        }
        else{
            log!(self.config.log_level,"[SEP] ended due to termination, evals/s: {}, iter/s: {}, took {:.3}s",FMT.fmt2(n_evals as f64 / start.elapsed().as_secs_f64()),FMT.fmt2(n_iter as f64 / start.elapsed().as_secs_f64()),start.elapsed().as_secs());
        }

        (min_overlap_sol.0, min_overlap_sol.1)
    }

    fn modify(&mut self) -> usize {
        let master_sol = self.prob.create_solution(None);

        let n_evals = self.workers.par_iter_mut()
            .map(|worker| {
                // Sync the workers with the master
                worker.load(&master_sol, &self.ot);
                // Let them modify
                worker.separate()
            })
            .sum();

        debug!("[MOD] optimizers w_o's: {:?}",self.workers.iter().map(|opt| opt.ot.get_total_weighted_overlap()).collect_vec());

        // Check which worker has the lowest total weighted overlap
        let best_opt = self.workers.iter_mut()
            .min_by_key(|opt| OrderedFloat(opt.ot.get_total_weighted_overlap()))
            .map(|opt| (opt.prob.create_solution(None), &opt.ot))
            .unwrap();

        // Sync the master with the best optimizer
        self.prob.restore_to_solution(&best_opt.0);
        self.ot = best_opt.1.clone();

        n_evals
    }

    pub fn rollback(&mut self, sol: &Solution, ots: Option<&OTSnapshot>) {
        debug_assert!(strip_width(sol) == self.prob.strip_width());
        self.prob.restore_to_solution(sol);

        match ots {
            Some(ots) => {
                //if a snapshot of the overlap tracker was provided, restore it
                self.ot.restore_but_keep_weights(ots, &self.prob.layout);
            }
            None => {
                //otherwise, rebuild it
                self.ot = OverlapTracker::new(&self.prob.layout);
            }
        }
    }

    pub fn move_item(&mut self, pk: PItemKey, d_transf: DTransformation) -> PItemKey {
        debug_assert!(tracker_matches_layout(&self.ot, &self.prob.layout));

        let item_id = self.prob.layout.placed_items()[pk].item_id;

        let old_overlap = self.ot.get_overlap(pk);
        let old_weighted_overlap = self.ot.get_weighted_overlap(pk);

        //Remove the item from the problem
        self.prob.remove_item(STRIP_LAYOUT_IDX, pk, true);

        //Place the item again but with a new transformation
        let (_, new_pk) = self.prob.place_item(PlacingOption {
            d_transf,
            layout_idx: STRIP_LAYOUT_IDX,
            item_id,
        });

        self.ot.register_item_move(&self.prob.layout, pk, new_pk);

        let new_overlap = self.ot.get_overlap(new_pk);
        let new_weighted_overlap = self.ot.get_weighted_overlap(new_pk);

        debug!("[MV] moved item {} from from o: {}, wo: {} to o+1: {}, w_o+1: {}",item_id,FMT.fmt2(old_overlap),FMT.fmt2(old_weighted_overlap),FMT.fmt2(new_overlap),FMT.fmt2(new_weighted_overlap));

        debug_assert!(tracker_matches_layout(&self.ot, &self.prob.layout));

        new_pk
    }

    pub fn change_strip_width(&mut self, new_width: f32, split_position: Option<f32>) {
        //if no split position is provided, use the center of the strip
        let split_position = split_position.unwrap_or(self.prob.strip_width() / 2.0);
        let delta = new_width - self.prob.strip_width();

        //shift all items right of the split position
        let items_to_shift = self.prob.layout.placed_items().iter()
            .filter(|(_, pi)| pi.shape.centroid().0 > split_position)
            .map(|(k, pi)| (k, pi.d_transf))
            .collect_vec();

        for (pik, dtransf) in items_to_shift {
            let existing_transf = dtransf.compose();
            let new_transf = existing_transf.translate((delta, 0.0));
            self.move_item(pik, new_transf.decompose());
        }

        //swap the bin to one with the new width
        let new_bin = Bin::from_strip(
            AARectangle::new(0.0, 0.0, new_width, self.prob.strip_height()),
            self.prob.layout.bin.base_cde.config().clone(),
        );
        self.prob.layout.change_bin(new_bin);

        //rebuild the overlap tracker
        self.ot = OverlapTracker::new(&self.prob.layout);

        //rebuild the workers
        self.workers.iter_mut().for_each(|opt| {
            *opt = SeparatorWorker {
                instance: self.instance.clone(),
                prob: self.prob.clone(),
                ot: self.ot.clone(),
                rng: SmallRng::seed_from_u64(self.rng.random()),
                sample_config: self.config.sample_config.clone(),
            };
        });
        debug!("[SEP] changed strip width to {:.3}", new_width);
    }

    pub fn export_svg(&mut self, solution: Option<Solution>, suffix: &str, only_live: bool) {
        if !EXPORT_ONLY_FINAL_SVG {
            if self.svg_counter == 0 {
                std::fs::create_dir_all(&self.output_svg_folder).unwrap();
                //remove all svg files from the directory. ONLY SVG FILES
                for file in std::fs::read_dir(&self.output_svg_folder).unwrap().flatten() {
                    if file.path().extension().unwrap_or_default() == "svg" {
                        std::fs::remove_file(file.path()).unwrap();
                    }
                }
            }

            let file_name = format!("{}_{:.3}_{suffix}", self.svg_counter, self.prob.strip_width());
            let svg = match solution {
                Some(sol) => s_layout_to_svg(&sol.layout_snapshots[0], &self.instance, DRAW_OPTIONS, file_name.as_str()),
                None => layout_to_svg(&self.prob.layout, &self.instance, DRAW_OPTIONS, file_name.as_str()),
            };

            if EXPORT_LIVE_SVG {
                if !Path::new(LIVE_DIR).exists() {
                    std::fs::create_dir_all(LIVE_DIR).unwrap();
                }
                io::write_svg(&svg, Path::new(&*format!("{}/.live_solution.svg", LIVE_DIR)), Level::Trace);
            }

            if !only_live {
                let file_path = &*format!("{}/{}.svg", &self.output_svg_folder, file_name);
                io::write_svg(&svg, Path::new(file_path), self.config.log_level);
                self.svg_counter += 1;
            }
        }
    }
}
