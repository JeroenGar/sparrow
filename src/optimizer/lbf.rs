use crate::eval::lbf_evaluator::LBFEvaluator;
use crate::eval::sample_eval::SampleEval;
use crate::sample::search::{search_placement, SampleConfig};
use itertools::Itertools;
use jagua_rs::entities::Instance;
use jagua_rs::probs::spp::entities::{SPInstance, SPPlacement, SPProblem};
use jagua_rs::Instant;
use log::debug;
use ordered_float::OrderedFloat;
use std::cmp::Reverse;
use std::iter;
use rand::rngs::Xoshiro256PlusPlus;

/// The initial-placement heuristic could not construct a solution.
/// This is not a proof that the instance is infeasible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConstructionError {
    pub item_id: usize,
}

impl std::fmt::Display for ConstructionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "could not construct an initial placement for item {}", self.item_id)
    }
}

impl std::error::Error for ConstructionError {}

pub struct LBFBuilder {
    pub instance: SPInstance,
    pub prob: SPProblem,
    pub rng: Xoshiro256PlusPlus,
    pub sample_config: SampleConfig,
}

impl LBFBuilder {
    pub fn new(
        instance: SPInstance,
        rng: Xoshiro256PlusPlus,
        sample_config: SampleConfig,
    ) -> Self {
        let prob = SPProblem::new(instance.clone());

        Self {
            instance,
            prob,
            rng,
            sample_config,
        }
    }

    /// Builds a complete initial placement.
    ///
    /// Returns an error if strip growth reaches the heuristic limit or cannot progress.
    pub fn construct(mut self) -> Result<Self, ConstructionError> {
        let start = Instant::now();
        let n_items = self.instance.items.len();
        let sorted_item_indices = (0..n_items)
            .sorted_by_cached_key(|id| {
                let item_shape = self.instance.item(*id).shape_cd.as_ref();
                let convex_hull_area = item_shape.surrogate().convex_hull_area;
                let diameter = item_shape.diameter;
                Reverse(OrderedFloat(convex_hull_area * diameter))
            })
            .flat_map(|id| {
                let missing_qty = self.prob.item_demand_qtys[id];
                iter::repeat_n(id, missing_qty)
            })
            .collect_vec();

        debug!("[CONSTR] placing items in order: {:?}",sorted_item_indices);

        for item_id in sorted_item_indices {
            self.place_item(item_id)?;
        }

        self.prob.fit_strip();
        debug!("[CONSTR] placed all items in width: {:.3} (in {:?})",self.prob.strip_width(), start.elapsed());
        Ok(self)
    }

    fn place_item(&mut self, item_id: usize) -> Result<(), ConstructionError> {
        loop {
            if let Some(placement) = self.find_placement(item_id) {
                self.prob.place_item(placement);
                debug!("[CONSTR] placing item {}/{} with id {} at [{}]", self.prob.layout.placed_items.len(), self.instance.total_item_qty(), placement.item_id, placement.d_transf);
                return Ok(());
            }

            let width = self.prob.strip_width();
            let next_width = width * 1.2;
            if !next_width.is_finite() || next_width <= width {
                return Err(ConstructionError { item_id });
            }
            // Retain the existing heuristic ceiling, without treating it as infeasibility.
            let width_limit = 2.0 * self.instance.items.iter()
                .map(|(item, qty)| item.shape_cd.diameter * *qty as f32)
                .sum::<f32>();
            if next_width >= width_limit {
                return Err(ConstructionError { item_id });
            }
            debug!("[CONSTR] failed to place item with id {}, expanding strip width", item_id);
            self.prob.change_strip_width(next_width);
        }
    }

    fn find_placement(&mut self, item_id: usize) -> Option<SPPlacement> {
        let layout = &self.prob.layout;
        let item = self.instance.item(item_id);
        let evaluator = LBFEvaluator::new(layout, item);

        let (best_sample, _) = search_placement(layout, item, None, evaluator, self.sample_config, &mut self.rng);

        match best_sample {
            Some((d_transf, SampleEval::Clear { .. })) => {
                Some(SPPlacement { item_id, d_transf })
            }
            _ => None
        }
    }
}
