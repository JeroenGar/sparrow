use std::iter::Sum;
use std::ops::AddAssign;
use crate::eval::separation_eval::SeparationEvaluator;
use crate::overlap::tracker::OverlapTracker;
use crate::sample::search;
use crate::sample::search::SampleConfig;
use crate::util::assertions::tracker_matches_layout;
use crate::FMT;
use itertools::Itertools;
use jagua_rs::entities::instances::instance_generic::InstanceGeneric;
use jagua_rs::entities::instances::strip_packing::SPInstance;
use jagua_rs::entities::placed_item::PItemKey;
use jagua_rs::entities::placing_option::PlacingOption;
use jagua_rs::entities::problems::problem_generic::{ProblemGeneric, STRIP_LAYOUT_IDX};
use jagua_rs::entities::problems::strip_packing::{strip_width, SPProblem};
use jagua_rs::entities::solution::Solution;
use jagua_rs::geometry::d_transformation::DTransformation;
use log::debug;
use rand::prelude::{SliceRandom, SmallRng};
use tap::Tap;

pub struct SeparatorWorker {
    pub instance: SPInstance,
    pub prob: SPProblem,
    pub ot: OverlapTracker,
    pub rng: SmallRng,
    pub sample_config: SampleConfig,
}

impl SeparatorWorker {
    pub fn load(&mut self, sol: &Solution, ot: &OverlapTracker) {
        // restores the state of the worker to the given solution and accompanying overlap tracker
        debug_assert!(strip_width(sol) == self.prob.strip_width());
        self.prob.restore_to_solution(sol);
        self.ot = ot.clone();
    }

    pub fn separate(&mut self) -> SepStats {
        //collect all overlapping items and shuffle them
        let candidates = self.prob.layout.placed_items().keys()
            .filter(|pk| self.ot.get_overlap(*pk) > 0.0)
            .collect_vec()
            .tap_mut(|v| v.shuffle(&mut self.rng));

        let mut total_moves = 0;
        let mut total_evals = 0;

        //give each item a chance to move to a better (less weighted overlapping) position
        for &pk in candidates.iter() {
            //check if the item is still overlapping
            if self.ot.get_overlap(pk) > 0.0 {
                let item_id = self.prob.layout.placed_items()[pk].item_id;
                let item = self.instance.item(item_id);

                // create an evaluator to evaluate the samples during the search
                let evaluator = SeparationEvaluator::new(&self.prob.layout, item, pk, &self.ot);

                // search for a better position for the item
                let (best_sample, n_evals) =
                    search::search_placement(&self.prob.layout, item, Some(pk), evaluator, self.sample_config, &mut self.rng);

                let (new_dt, _eval) = best_sample.expect("search_placement should always return a sample");

                // move the item to the new position
                self.move_item(pk, new_dt);
                total_moves += 1;
                total_evals += n_evals;
            }
        }
        SepStats { total_moves, total_evals }
    }

    pub fn move_item(&mut self, pk: PItemKey, d_transf: DTransformation) -> PItemKey {
        debug_assert!(tracker_matches_layout(&self.ot, &self.prob.layout));

        let item = self.instance.item(self.prob.layout.placed_items()[pk].item_id);

        let (old_o, old_w_o) = (self.ot.get_overlap(pk), self.ot.get_weighted_overlap(pk));

        //modify the problem, by removing the item and placing it in the new position
        self.prob.remove_item(STRIP_LAYOUT_IDX, pk, true);
        let (_, new_pk) = self.prob.place_item(
            PlacingOption {
                d_transf,
                item_id: item.id,
                layout_idx: STRIP_LAYOUT_IDX,
            }
        );
        //update the overlap tracker to reflect the changes
        self.ot.register_item_move(&self.prob.layout, pk, new_pk);

        let (new_o, new_w_o) = (self.ot.get_overlap(new_pk), self.ot.get_weighted_overlap(new_pk));

        debug!("Moved item {} from from o: {}, wo: {} to o+1: {}, w_o+1: {}",item.id,FMT.fmt2(old_o),FMT.fmt2(old_w_o),FMT.fmt2(new_o),FMT.fmt2(new_w_o));
        debug_assert!(tracker_matches_layout(&self.ot, &self.prob.layout));

        new_pk
    }
}

pub struct SepStats {
    pub total_moves: usize,
    pub total_evals: usize,
}

impl Sum for SepStats {
    fn sum<I: Iterator<Item=SepStats>>(iter: I) -> Self {
        let mut total_moves = 0;
        let mut total_evals = 0;

        for report in iter {
            total_moves += report.total_moves;
            total_evals += report.total_evals;
        }

        SepStats { total_moves, total_evals }
    }
}

impl AddAssign for SepStats {
    fn add_assign(&mut self, other: Self) {
        self.total_moves += other.total_moves;
        self.total_evals += other.total_evals;
    }
}
