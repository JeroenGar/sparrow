use crate::config::OVERLAP_PROXY_EPSILON_DIAM_RATIO;
use jagua_rs::geometry::fail_fast::sp_surrogate::SPSurrogate;
use jagua_rs::geometry::geo_traits::{Distance, Shape};
use jagua_rs::geometry::primitives::aa_rectangle::AARectangle;
use jagua_rs::geometry::primitives::simple_polygon::SimplePolygon;

/// Evaluates the overlap between two simple polygons.
#[inline(always)]
pub fn eval_overlap_poly_poly(s1: &SimplePolygon, s2: &SimplePolygon) -> f32 {
    let epsilon = f32::max(s1.diameter(), s2.diameter()) * OVERLAP_PROXY_EPSILON_DIAM_RATIO;

    let overlap_proxy = poles_overlap_area_proxy(&s1.surrogate(), &s2.surrogate(), epsilon) + epsilon.powi(2);

    debug_assert!(overlap_proxy.is_normal());

    overlap_proxy.sqrt() * calc_penalty(s1, s2)
}

pub fn calc_penalty(s1: &SimplePolygon, s2: &SimplePolygon) -> f32 {
    let p1 = s1.surrogate().convex_hull_area.sqrt();
    let p2 = s2.surrogate().convex_hull_area.sqrt();
    //harmonic mean
    2.0 * (p1 * p2) / (p1 + p2)
}

/// Evaluates the overlap between a simple polygon and the exterior of the bin.
#[inline(always)]
pub fn eval_overlap_poly_bin(s: &SimplePolygon, bin_bbox: AARectangle) -> f32 {
    let s_bbox = s.bbox();
    let overlap = match AARectangle::from_intersection(&s_bbox, &bin_bbox) {
        Some(r) => {
            //intersection exist, calculate the area of the intersection (+ a small value to ensure it is never zero)
            let negative_area = (s_bbox.area() - r.area()) + 0.001 * s_bbox.area();
            negative_area
        }
        None => {
            //no intersection, guide towards intersection with bin
            s_bbox.area() + s_bbox.centroid().distance(&bin_bbox.centroid())
        }
    };
    debug_assert!(overlap.is_normal());

    let penalty = s.surrogate().convex_hull_area;

    10.0 * (overlap * penalty).sqrt()
}


/// Calculates a proxy for the overlap area between two sets of poles
#[inline(always)]
pub fn poles_overlap_area_proxy<'a>(sp1: &SPSurrogate, sp2: &SPSurrogate, epsilon: f32) -> f32 {
    let mut total_overlap = 0.0;
    for p1 in &sp1.poles {
        for p2 in &sp2.poles {
            // Penetration depth between the two poles (circles)
            let pd = (p1.radius + p2.radius) - p1.center.distance(&p2.center);

            let pd_decay = match pd >= epsilon {
                true => pd,
                false => epsilon.powi(2) / (-pd + 2.0 * epsilon),
            };

            total_overlap += pd_decay * f32::min(p1.radius, p2.radius);
        }
    }
    debug_assert!(total_overlap.is_normal());
    total_overlap
}