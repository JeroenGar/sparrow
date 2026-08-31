use crate::quantify::quantify_collision_poly_container;
#[cfg(not(feature = "simd"))]
use crate::quantify::quantify_collision_poly_poly;
#[cfg(feature = "simd")]
use crate::quantify::simd::circles_soa::CirclesSoA;
#[cfg(feature = "simd")]
use crate::quantify::simd::quantify_collision_poly_poly_simd_bounded;
use crate::quantify::tracker::CollisionTracker;
use jagua_rs::collision_detection::hazards::HazardEntity;
use jagua_rs::entities::{Layout, PItemKey};
use jagua_rs::geometry::primitives::SPolygon;

/// Computes Sparrow's collision loss as `jagua-rs` discovers hazards.
pub(super) struct CollisionLossEvaluator<'a> {
    layout: &'a Layout,
    ct: &'a CollisionTracker,
    current_pk: PItemKey,
    loss: f32,
    loss_bound: f32,
    #[cfg(feature = "simd")]
    poles_soa: CirclesSoA,
}

impl<'a> CollisionLossEvaluator<'a> {
    pub(super) fn new(layout: &'a Layout, ct: &'a CollisionTracker, current_pk: PItemKey) -> Self {
        Self {
            layout,
            ct,
            current_pk,
            loss: 0.0,
            loss_bound: f32::INFINITY,
            #[cfg(feature = "simd")]
            poles_soa: CirclesSoA::new(),
        }
    }

    pub(super) fn reload(&mut self, loss_bound: f32, _shape: &SPolygon) {
        self.loss = 0.0;
        self.loss_bound = loss_bound;
        #[cfg(feature = "simd")]
        self.poles_soa.load(&_shape.surrogate().poles);
    }

    pub(super) fn add(&mut self, hazard: HazardEntity, shape: &SPolygon) -> bool {
        let remaining = self.loss_bound - self.loss;
        let Some(extra_loss) = self.calc_weighted_loss_bounded(hazard, shape, remaining) else {
            return true;
        };
        self.loss += extra_loss;
        self.loss > self.loss_bound
    }

    pub(super) fn loss(&self) -> f32 {
        self.loss
    }

    fn calc_weighted_loss_bounded(
        &self,
        hazard: HazardEntity,
        shape: &SPolygon,
        max_loss: f32,
    ) -> Option<f32> {
        match hazard {
            HazardEntity::PlacedItem { pk: other_pk, .. } => {
                let other_shape = &self.layout.placed_items[other_pk].shape;
                let weight = self.ct.get_pair_weight(self.current_pk, other_pk);

                #[cfg(feature = "simd")]
                {
                    quantify_collision_poly_poly_simd_bounded(
                        other_shape,
                        shape,
                        &self.poles_soa,
                        max_loss / weight,
                    )
                    .map(|loss| loss * weight)
                }

                #[cfg(not(feature = "simd"))]
                {
                    let loss = quantify_collision_poly_poly(other_shape, shape) * weight;
                    (loss <= max_loss).then_some(loss)
                }
            }
            HazardEntity::Exterior => Some(
                quantify_collision_poly_container(shape, self.layout.container.outer_cd.bbox)
                    * self.ct.get_container_weight(self.current_pk),
            ),
            _ => unimplemented!("unsupported hazard entity"),
        }
    }
}
