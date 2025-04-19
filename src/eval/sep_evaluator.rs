use crate::eval::sample_eval::{SampleEval, SampleEvaluator};
use crate::eval::specialized_jaguars_pipeline::{collect_poly_collisions_in_detector_custom, SpecializedHazardDetector};
use crate::quantify::tracker::CollisionTracker;
use jagua_rs::collision_detection::hazards::detector::HazardDetector;
use jagua_rs::entities::general::Item;
use jagua_rs::entities::general::Layout;
use jagua_rs::entities::general::PItemKey;
use jagua_rs::geometry::DTransformation;
use jagua_rs::geometry::primitives::SPolygon;

pub struct SeparationEvaluator<'a> {
    layout: &'a Layout,
    item: &'a Item,
    detection_map: SpecializedHazardDetector<'a>,
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
        let detection_map = SpecializedHazardDetector::new(layout, ct, current_pk);

        Self {
            layout,
            item,
            detection_map,
            shape_buff: item.shape_cd.as_ref().clone(),
            n_evals: 0,
        }
    }
}

impl<'a> SampleEvaluator for SeparationEvaluator<'a> {
    /// Evaluates a transformation. An upper bound can be provided to early terminate the process.
    fn eval(&mut self, dt: DTransformation, upper_bound: Option<SampleEval>) -> SampleEval {
        self.n_evals += 1;
        let cde = self.layout.cde();

        // evals with higher loss than this will always be rejected
        let loss_bound = match upper_bound {
            Some(SampleEval::Collision { loss }) => loss,
            Some(SampleEval::Clear { .. }) => 0.0,
            _ => f32::INFINITY,
        };
        // reload the detection map for the new query and update the loss bound
        self.detection_map.reload(loss_bound);

        // Query the CDE, all colliding hazards will be stored in the detection map
        collect_poly_collisions_in_detector_custom(cde, &dt, &mut self.shape_buff, self.item.shape_cd.as_ref(), &mut self.detection_map);

        if self.detection_map.early_terminate(&self.shape_buff) {
            //the detection map is in early termination state, this means potentially not all collisions were detected,
            //but its loss was above the loss bound anyway
            SampleEval::Invalid
        } else if self.detection_map.is_empty() {
            SampleEval::Clear { loss: 0.0 }
        } else {
            SampleEval::Collision {
                loss: self.detection_map.loss(&self.shape_buff),
            }
        }
    }

    fn n_evals(&self) -> usize {
        self.n_evals
    }
}

