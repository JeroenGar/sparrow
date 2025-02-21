use itertools::Itertools;
use jagua_rs::entities::instances::instance_generic::InstanceGeneric;
use jagua_rs::entities::instances::strip_packing::SPInstance;
use jagua_rs::entities::placed_item::PItemKey;
use jagua_rs::entities::placing_option::PlacingOption;
use jagua_rs::entities::problems::problem_generic::{ProblemGeneric, STRIP_LAYOUT_IDX};
use jagua_rs::entities::problems::strip_packing::SPProblem;
use jagua_rs::entities::solution::Solution;
use jagua_rs::fsize;
use jagua_rs::geometry::d_transformation::DTransformation;
use jagua_rs::geometry::geo_enums::GeoRelation;
use jagua_rs::geometry::geo_traits::{Shape, Transformable};
use log::debug;
use rand::prelude::{SliceRandom, SmallRng};
use tap::Tap;
use crate::FMT;
use crate::opt::gls_orchestrator;
use crate::overlap::overlap_tracker;
use crate::overlap::overlap_tracker::OverlapTracker;
use crate::sample::eval::overlapping_evaluator::OverlappingSampleEvaluator;
use crate::sample::eval::SampleEval;
use crate::sample::search;
use crate::sample::search::SearchConfig;

pub struct GLSWorker {
    pub instance: SPInstance,
    pub prob: SPProblem,
    pub ot: OverlapTracker,
    pub rng: SmallRng,
    pub ch_area_cutoff: fsize,
}

impl GLSWorker {
    pub fn load(&mut self, sol: &Solution, ot: &OverlapTracker) {
        self.prob.restore_to_solution(sol);
        self.ot = ot.clone();
    }

    pub fn separate(&mut self) -> usize {
        let candidates = self.prob.layout.placed_items()
            .keys()
            .filter(|pk| self.ot.get_overlap(*pk) > 0.0)
            .collect_vec()
            .tap_mut(|v| v.shuffle(&mut self.rng));

        let mut n_movements = 0;

        for &pk in candidates.iter() {
            let current_overlap = self.ot.get_overlap(pk);
            if current_overlap > 0.0 {
                let item = self.instance.item(self.prob.layout.placed_items()[pk].item_id);

                let evaluator = OverlappingSampleEvaluator::new(
                    &self.prob.layout,
                    item,
                    Some(pk),
                    &self.ot,
                );

                let new_placement = search::search_placement(
                    &self.prob.layout,
                    item,
                    Some(pk),
                    evaluator,
                    generate_search_config(&self.ot, pk),
                    &mut self.rng,
                );

                self.move_item(pk, new_placement.0, Some(new_placement.1));
                n_movements += 1;
            }
        }
        n_movements
    }

    pub fn move_item(&mut self, pik: PItemKey, d_transf: DTransformation, eval: Option<SampleEval>) -> PItemKey {
        debug_assert!(overlap_tracker::tracker_matches_layout(&self.ot, &self.prob.layout));

        let old_overlap = self.ot.get_overlap(pik);
        let old_weighted_overlap = self.ot.get_weighted_overlap(pik);
        let old_bbox = self.prob.layout.placed_items()[pik].shape.bbox();

        //Remove the item from the problem
        let old_p_opt = self.prob.remove_item(STRIP_LAYOUT_IDX, pik, true);
        let item = self.instance.item(old_p_opt.item_id);

        //Compute the colliding entities after the move
        let colliding_entities = {
            let shape = item.shape.clone().as_ref().clone()
                .tap_mut(|s| { s.transform(&d_transf.compose()); });

            let mut colliding_entities = vec![];
            self.prob.layout.cde().collect_poly_collisions(&shape, &[], &mut colliding_entities);
            colliding_entities
        };

        assert!(colliding_entities.is_empty() || !matches!(eval, Some(SampleEval::Valid(_))), "colliding entities detected for valid placement");

        let new_pk = {
            let new_p_opt = PlacingOption {
                d_transf,
                ..old_p_opt
            };

            let (_, new_pik) = self.prob.place_item(new_p_opt);
            new_pik
        };

        self.ot.register_item_move(&self.prob.layout, pik, new_pk);

        let new_overlap = self.ot.get_overlap(new_pk);
        let new_weighted_overlap = self.ot.get_weighted_overlap(new_pk);
        let new_bbox = self.prob.layout.placed_items()[new_pk].shape.bbox();

        let jumped = old_bbox.relation_to(&new_bbox) == GeoRelation::Disjoint;
        let item_big_enough = item.shape.surrogate().convex_hull_area > self.ch_area_cutoff;
        if jumped && item_big_enough {
            self.ot.register_jump(new_pk);
        }

        debug!("Moved item {} from from o: {}, wo: {} to o+1: {}, w_o+1: {} (jump: {})", item.id, FMT.fmt2(old_overlap), FMT.fmt2(old_weighted_overlap), FMT.fmt2(new_overlap), FMT.fmt2(new_weighted_overlap), jumped);

        debug_assert!(overlap_tracker::tracker_matches_layout(&self.ot, &self.prob.layout));

        new_pk
    }
}

pub fn generate_search_config(ot: &OverlapTracker, pk: PItemKey) -> SearchConfig {
    let on_jump_cooldown = ot.is_on_jump_cooldown(pk);
    match on_jump_cooldown {
        false => SearchConfig {
            n_bin_samples: gls_orchestrator::N_UNIFORM_SAMPLES / 2,
            n_focussed_samples: gls_orchestrator::N_UNIFORM_SAMPLES / 2,
            n_coord_descents: gls_orchestrator::N_COORD_DESCENTS,
        },
        true => SearchConfig {
            n_bin_samples: 0,
            n_focussed_samples: gls_orchestrator::N_UNIFORM_SAMPLES,
            n_coord_descents: gls_orchestrator::N_COORD_DESCENTS,
        }
    }
}