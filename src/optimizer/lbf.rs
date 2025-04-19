use crate::eval::lbf_evaluator::LBFEvaluator;
use crate::eval::sample_eval::SampleEval;
use crate::sample::search::{search_placement, SampleConfig};
use itertools::Itertools;
use log::debug;
use ordered_float::OrderedFloat;
use rand::prelude::SmallRng;
use std::cmp::Reverse;
use std::iter;
use std::time::Instant;
use jagua_rs::collision_detection::CDEConfig;
use jagua_rs::entities::general::Instance;
use jagua_rs::entities::strip_packing::{SPInstance, SPPlacement, SPProblem};

pub struct LBFBuilder {
    pub instance: SPInstance,
    pub prob: SPProblem,
    pub rng: SmallRng,
    pub sample_config: SampleConfig,
}

impl LBFBuilder {
    pub fn new(
        instance: SPInstance,
        cde_config: CDEConfig,
        rng: SmallRng,
        sample_config: SampleConfig,
    ) -> Self {
        let init_strip_width = instance.item_area / instance.strip_height; //100% utilization
        let prob = SPProblem::new(instance.clone(), init_strip_width, cde_config);

        Self {
            instance,
            prob,
            rng,
            sample_config,
        }
    }

    pub fn construct(mut self) -> Self {
        let start = Instant::now();
        let n_items = self.instance.items().len();
        let sorted_item_indices = (0..n_items)
            .sorted_by_cached_key(|id| {
                let item_shape = self.instance.items()[*id].0.shape_cd.as_ref();
                let convex_hull_area = item_shape.surrogate().convex_hull_area;
                let diameter = item_shape.diameter;
                Reverse(OrderedFloat(convex_hull_area * diameter))
            })
            .map(|id| {
                let missing_qty = self.prob.missing_item_qtys[id].max(0) as usize;
                iter::repeat(id).take(missing_qty)
            })
            .flatten()
            .collect_vec();

        debug!("[CONSTR] placing items in order: {:?}",sorted_item_indices);

        for item_id in sorted_item_indices {
            self.place_item(item_id);
        }

        self.prob.fit_strip();
        debug!("[CONSTR] placed all items in width: {:.3} (in {:?})",self.prob.strip_width(), start.elapsed());
        self
    }

    fn place_item(&mut self, item_id: usize) {
        match self.find_placement(item_id) {
            Some(p_opt) => {
                self.prob.place_item(p_opt);
                debug!("[CONSTR] placing item {}/{} with id {} at [{}]",self.prob.layout.placed_items.len(),self.instance.total_item_qty(),p_opt.item_id,p_opt.d_transf);
            }
            None => {
                debug!("[CONSTR] failed to place item with id {}, expanding strip width",item_id);
                self.prob.change_strip_width(self.prob.strip_width() * 1.2);
                self.place_item(item_id);
            }
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
