use jagua_rs::collision_detection::hazards::collector::HazardCollector;
use crate::eval::sample_eval::{SampleEval, SampleEvaluator};
use crate::eval::specialized_jaguars_pipeline::{collect_poly_collisions_in_detector_custom, SpecializedHazardCollector};
use crate::quantify::tracker::CollisionTracker;
use jagua_rs::entities::Item;
use jagua_rs::entities::Layout;
use jagua_rs::entities::PItemKey;
use jagua_rs::geometry::DTransformation;
use jagua_rs::geometry::primitives::SPolygon;

pub struct SeparationEvaluator<'a> {
    layout: &'a Layout,
    item: &'a Item,
    collector: SpecializedHazardCollector<'a>,
    shape_buff: SPolygon,
    n_evals: usize,
}

impl<'a> SeparationEvaluator<'a> {
    pub fn new(
        layout: &'a Layout,
        item: &'a Item,
        current_pk: PItemKey,
        ct: &'a CollisionTracker,
    ) -> Self {
        let collector = SpecializedHazardCollector::new(layout, ct, current_pk);

        Self {
            layout,
            item,
            collector,
            shape_buff: item.shape_cd.as_ref().clone(),
            n_evals: 0,
        }
    }
}

impl<'a> SampleEvaluator for SeparationEvaluator<'a> {
    /// Evaluates a transformation. An upper bound can be provided to early terminate the process.
    /// Algorithm 7 from https://doi.org/10.48550/arXiv.2509.13329
    fn evaluate_sample(&mut self, dt: DTransformation, upper_bound: Option<SampleEval>) -> SampleEval {
        self.n_evals += 1;
        let cde = self.layout.cde();

        //samples which evaluate higher than this will certainly be rejected
        let loss_bound = match upper_bound {
            Some(SampleEval::Collision { loss }) => loss,
            Some(SampleEval::Clear { .. }) => 0.0,
            _ => f32::INFINITY,
        };
        //reload the hazard collector to prepare for a new query
        self.collector.reload(loss_bound);

        //query the CDE, all colliding hazards will be stored in the detection map
        collect_poly_collisions_in_detector_custom(cde, &dt, &mut self.shape_buff, self.item.shape_cd.as_ref(), &mut self.collector);

        if self.collector.early_terminate(&self.shape_buff) {
            //the detection map is in early termination state, this means potentially not all collisions were detected,
            //but its loss was above the loss bound anyway
            SampleEval::Invalid
        } else if self.collector.is_empty() {
            SampleEval::Clear { loss: 0.0 }
        } else {
            SampleEval::Collision {
                loss: self.collector.loss(&self.shape_buff),
            }
        }
    }

    fn n_evals(&self) -> usize {
        self.n_evals
    }
}

