use crate::quantify::quantify_collision_poly_bin;
#[cfg(not(feature = "simd"))]
use crate::quantify::quantify_collision_poly_poly;
#[cfg(feature = "simd")]
use crate::quantify::simd::circles_soa::CirclesSoA;
#[cfg(feature = "simd")]
use crate::quantify::simd::quantify_collision_poly_poly_simd;
use crate::quantify::tracker::CollisionTracker;
use crate::util::assertions;
use crate::util::bit_reversal_iterator::BitReversalIterator;
use float_cmp::approx_eq;
use jagua_rs::collision_detection::CDEngine;
use jagua_rs::collision_detection::hazards::HazardEntity;
use jagua_rs::collision_detection::hazards::detector::HazardDetector;
use jagua_rs::collision_detection::quadtree::{QTHazPresence, QTQueryable};
use jagua_rs::collision_detection::quadtree::QTNode;
use jagua_rs::entities::general::Layout;
use jagua_rs::entities::general::PItemKey;
use jagua_rs::geometry::DTransformation;
use jagua_rs::geometry::geo_traits::{CollidesWith, Shape, TransformableFrom};
use jagua_rs::geometry::primitives::SPolygon;
use slotmap::SecondaryMap;

/// Functionally the same as [`CDEngine::collect_poly_collisions_in_detector`], but with early termination.
/// Saving quite a bit of CPU time since over 90% of the time is spent in this function.
pub fn collect_poly_collisions_in_detector_custom(
    cde: &CDEngine,
    dt: &DTransformation,
    shape_buffer: &mut SPolygon,
    reference_shape: &SPolygon,
    det: &mut SpecializedHazardDetector,
) {
    let t = dt.compose();
    // transform the shape buffer to the new position
    shape_buffer.transform_from(reference_shape, &t);
    let shape = shape_buffer;

    #[cfg(feature = "simd")]
    det.poles_soa.load(&shape.surrogate().poles);

    // Start off by checking a few poles to detect obvious collisions quickly
    for pole in shape.surrogate().ff_poles() {
        qt_collect_collisions_custom(&cde.quadtree, pole, det);
        if det.early_terminate(shape) { return; }
    }

    // Collect collisions for all edges of the shape.
    // Iterate over them in a bit-reversed order to maximize detecting new hazards early.
    let custom_edge_iter = BitReversalIterator::new(shape.n_vertices())
        .map(|i| shape.edge(i));
    for edge in custom_edge_iter {
        qt_collect_collisions_custom(&cde.quadtree, &edge, det);
        if det.early_terminate(shape) { return; }
    }

    // At this point, all collisions due to edge-edge intersection are detected.
    // The only type of collisions that possibly remain is containment.

    let checkpoint = det.idx_counter;

    // Detect all potential hazards within the bounding box of the shape.
    cde.collect_potential_hazards_within(&shape.bbox(), det);

    if det.idx_counter > checkpoint {
        // Additional hazards were detected, check if they are contained in each other.
        // If they are not, remove them again from the detector, as they do not collide with the shape
        for haz in cde.all_hazards().filter(|h| h.active) {
            match haz.entity {
                HazardEntity::BinExterior => {
                    if let Some((_, idx)) = det.detected_bin {
                        if idx >= checkpoint {
                            // If the bin was detected as a potential containment, remove it.
                            // For this specific problem, an item can never be entirely outside the bin (rectangle).
                            det.remove(&haz.entity)
                        }
                    }
                }
                HazardEntity::PlacedItem { pk, .. } => {
                    if let Some((_, idx)) = det.detected_pis.get(pk) {
                        if *idx >= checkpoint {
                            // The item was not detected during the quadtree query, but was detected as a potential containment.
                            if !cde.poly_or_hazard_are_contained(shape, haz) {
                                //The item is not contained in the shape, remove it from the detector
                                det.remove(&haz.entity)
                            }
                        }
                    }
                }
                _ => unreachable!("unsupported hazard entity"),
            }
        }
    }
    // At this point, all collisions should be present in the detector.
    debug_assert!(assertions::custom_pipeline_matches_jaguars(shape, det));
}

/// Modified version of [`jagua_rs::collision_detection::hazard_helpers::DetectionMap`]
/// This struct computes the loss incrementally on the fly and caches the result.
/// Allows for early termination if the loss exceeds a certain upperbound.
pub struct SpecializedHazardDetector<'a> {
    pub layout: &'a Layout,
    pub ct: &'a CollisionTracker,
    pub current_pk: PItemKey,
    pub detected_pis: SecondaryMap<PItemKey, (HazardEntity, usize)>,
    pub detected_bin: Option<(HazardEntity, usize)>,
    pub idx_counter: usize,
    pub loss_cache: (usize, f32),
    pub loss_bound: f32,
    #[cfg(feature = "simd")]
    pub poles_soa: CirclesSoA,
}

impl<'a> SpecializedHazardDetector<'a> {
    pub fn new(
        layout: &'a Layout,
        ct: &'a CollisionTracker,
        current_pk: PItemKey,
    ) -> Self {
        Self {
            layout,
            ct,
            current_pk,
            detected_pis: SecondaryMap::new(),
            detected_bin: None,
            idx_counter: 0,
            loss_cache: (0, 0.0),
            loss_bound: f32::INFINITY,
            #[cfg(feature = "simd")]
            poles_soa: CirclesSoA::new(),
        }
    }

    pub fn reload(&mut self, loss_bound: f32) {
        self.detected_pis.clear();
        self.detected_bin = None;
        self.idx_counter = 0;
        self.loss_cache = (0, 0.0);
        self.loss_bound = loss_bound;
    }

    pub fn iter_with_index(&self) -> impl Iterator<Item=&(HazardEntity, usize)> {
        self.detected_pis.values().chain(self.detected_bin.iter())
    }

    pub fn early_terminate(&mut self, shape: &SPolygon) -> bool {
        self.loss(shape) > self.loss_bound
    }

    pub fn loss(&mut self, shape: &SPolygon) -> f32 {
        let (cache_idx, cached_loss) = self.loss_cache;
        if cache_idx < self.idx_counter {
            // additional hazards were detected, update the cache
            let extra_loss: f32 = self.iter_with_index()
                .filter(|(_, idx)| *idx >= cache_idx)
                .map(|(h, _)| self.calc_weighted_loss(h, shape))
                .sum();
            self.loss_cache = (self.idx_counter, cached_loss + extra_loss);
        }
        debug_assert!(approx_eq!(f32, self.loss_cache.1, self.iter().map(|h| self.calc_weighted_loss(h, shape)).sum()));
        self.loss_cache.1
    }

    fn calc_weighted_loss(&self, haz: &HazardEntity, shape: &SPolygon) -> f32 {
        match haz {
            HazardEntity::PlacedItem { pk: other_pk, .. } => {
                let other_shape = &self.layout.placed_items[*other_pk].shape;

                #[cfg(not(feature = "simd"))]
                let loss = quantify_collision_poly_poly(other_shape, shape);
                #[cfg(feature = "simd")]
                let loss = quantify_collision_poly_poly_simd(other_shape, shape, &self.poles_soa);

                let weight = self.ct.get_pair_weight(self.current_pk, *other_pk);
                loss * weight
            }
            HazardEntity::BinExterior => {
                let loss = quantify_collision_poly_bin(shape, self.layout.bin.outer_cd.bbox());
                let weight = self.ct.get_bin_weight(self.current_pk);
                2.0 * loss * weight
            }
            _ => unimplemented!("unsupported hazard entity"),
        }
    }
}

impl<'a> HazardDetector for SpecializedHazardDetector<'a> {
    fn contains(&self, haz: &HazardEntity) -> bool {
        match haz {
            HazardEntity::PlacedItem { pk, .. } => {
                *pk == self.current_pk || self.detected_pis.contains_key(*pk)
            }
            HazardEntity::BinExterior => self.detected_bin.is_some(),
            _ => unreachable!("unsupported hazard entity"),
        }
    }

    fn push(&mut self, haz: HazardEntity) {
        debug_assert!(!self.contains(&haz));
        match haz {
            HazardEntity::PlacedItem { pk, .. } => {
                self.detected_pis.insert(pk, (haz, self.idx_counter));
            }
            HazardEntity::BinExterior => {
                self.detected_bin = Some((HazardEntity::BinExterior, self.idx_counter))
            }
            _ => unreachable!("unsupported hazard entity"),
        }
        self.idx_counter += 1;
    }

    fn remove(&mut self, haz: &HazardEntity) {
        match haz {
            HazardEntity::PlacedItem { pk, .. } => {
                let (_, idx) = self.detected_pis.remove(*pk).unwrap();
                if idx < self.loss_cache.0 {
                    //wipe the cache if a hazard was removed that was in it
                    self.loss_cache = (0, 0.0);
                }
            }
            HazardEntity::BinExterior => {
                let (_, idx) = self.detected_bin.take().unwrap();
                if idx < self.loss_cache.0 {
                    //wipe the cache if a hazard was removed that was in it
                    self.loss_cache = (0, 0.0);
                }
            }
            _ => unreachable!("unsupported hazard entity"),
        }
    }

    fn len(&self) -> usize {
        self.detected_pis.len() + self.detected_bin.is_some() as usize
    }

    fn iter(&self) -> impl Iterator<Item=&HazardEntity> {
        self.detected_pis.iter().map(|(_, (h, _))| h)
            .chain(self.detected_bin.iter().map(|(h, _)| h))
    }
}

/// Mirrors [`QTNode::collect_collisions`] but slightly faster for this specific use case.
pub fn qt_collect_collisions_custom<T: QTQueryable>(qtn: &QTNode, entity: &T, detector: &mut SpecializedHazardDetector) {
    match entity.collides_with(&qtn.bbox) {
        false => return, //Entity does not collide with the node
        true => match qtn.children.as_ref() {
            Some(children) => {
                //Do not perform any CD on this level, check the children
                children.iter().for_each(|child| qt_collect_collisions_custom(child, entity, detector));
            }
            None => {
                //No children, detect all Entire hazards and check the Partial ones
                for hz in qtn.hazards.active_hazards().iter() {
                    match &hz.presence {
                        QTHazPresence::None => (),
                        QTHazPresence::Entire => {
                            if !detector.contains(&hz.entity) {
                                detector.push(hz.entity)
                            }
                        }
                        QTHazPresence::Partial(p_haz) => {
                            if !detector.contains(&hz.entity) {
                                if p_haz.collides_with(entity) {
                                    detector.push(hz.entity);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}