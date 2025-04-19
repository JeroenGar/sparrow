use crate::config::OVERLAP_PROXY_EPSILON_DIAM_RATIO;
use crate::quantify::calc_shape_penalty;
use crate::quantify::simd::circles_soa::CirclesSoA;
use crate::quantify::simd::overlap_proxy_simd::poles_overlap_area_proxy_simd;
use jagua_rs::geometry::geo_traits::Shape;
use jagua_rs::geometry::primitives::SPolygon;

pub mod circles_soa;
pub mod overlap_proxy_simd;


/// Quantifies a collision between two simple polygons using SIMD.
/// Mirrors the functionality of `quantify_collision_poly_poly` but leverages SIMD instructions.
#[inline(always)]
pub fn quantify_collision_poly_poly_simd(s1: &SPolygon, s2: &SPolygon, poles2: &CirclesSoA) -> f32 {
    let epsilon = f32::max(s1.diameter(), s2.diameter()) * OVERLAP_PROXY_EPSILON_DIAM_RATIO;

    let overlap_proxy = poles_overlap_area_proxy_simd(&s1.surrogate(), &s2.surrogate(), epsilon, poles2) + epsilon.powi(2);

    debug_assert!(overlap_proxy.is_normal());

    let penalty = calc_shape_penalty(s1, s2);

    overlap_proxy.sqrt() * penalty
}