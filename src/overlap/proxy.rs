use crate::config::OVERLAP_PROXY_EPSILON_DIAM_RATIO;
use jagua_rs::geometry::fail_fast::sp_surrogate::SPSurrogate;
use jagua_rs::geometry::geo_traits::{Distance, Shape};
use jagua_rs::geometry::primitives::aa_rectangle::AARectangle;
use jagua_rs::geometry::primitives::simple_polygon::SimplePolygon;
use ordered_float::NotNan;
use std::cmp::Ordering;

#[inline(always)]
pub fn poly_overlap_proxy(s1: &SimplePolygon, s2: &SimplePolygon) -> f32 {
    let epsilon = f32::max(s1.diameter(), s2.diameter()) * OVERLAP_PROXY_EPSILON_DIAM_RATIO;

    let deficit = poles_overlap_proxy(&s1.surrogate(), &s2.surrogate(), epsilon);

    debug_assert!(
        deficit > 0.0,
        "d:{deficit} has to be greater than 0.0. safety margins: {}, {}",
        s1.surrogate().max_distance_point_to_pole,
        s2.surrogate().max_distance_point_to_pole
    );

    let s1_penalty = s1.surrogate().convex_hull_area;
    let s2_penalty = s2.surrogate().convex_hull_area;

    let penalty = 0.75 * f32::min(s1_penalty, s2_penalty) + 0.25 * f32::max(s1_penalty, s2_penalty);

    (deficit * penalty).sqrt()
}

#[inline(always)]
pub fn bin_overlap_proxy(s: &SimplePolygon, bin_bbox: AARectangle) -> f32 {
    let s_bbox = s.bbox();
    let deficit = match AARectangle::from_intersection(&s_bbox, &bin_bbox) {
        Some(r) => {
            let negative_area = s_bbox.area() - r.area();
            negative_area
        }
        None => {
            //no intersection, guide towards intersection with bin
            s_bbox.area() + s_bbox.centroid().distance(&bin_bbox.centroid())
        }
    };
    let penalty = s.surrogate().convex_hull_area;

    10.0 * (deficit * penalty).sqrt()
}

#[inline(always)]
pub fn poles_overlap_proxy<'a>(sp1: &SPSurrogate, sp2: &SPSurrogate, epsilon: f32) -> f32 {
    let (sp_inner, sp_outer) = choose_inner_outer(sp1, sp2);
    let bpole_inner = sp_inner.poles_bounding_circle.clone();
    let safety_margin = sp1.max_distance_point_to_pole + sp2.max_distance_point_to_pole;

    let mut total_deficit = 0.0;
    for p_o in &sp_outer.poles {
        //if the pole is far enough outside the bounding circle of the inner surrogate poles, skip it.
        //its deficit will be negligible and this speeds up the calculation quite a bit
        let sq_distance = p_o.center.sq_distance(&bpole_inner.center);
        let neglect_sq_dist = (p_o.radius + bpole_inner.radius + safety_margin).powi(2);
        if sq_distance > neglect_sq_dist {
            //p_o is far enough away from the inner surrogate's bounding pole that it can be neglected,
            //saving precious time
            continue;
        } else {
            for p_i in &sp_inner.poles {
                let pd = (p_o.radius + p_i.radius) - p_o.center.distance(&p_i.center);

                let pd_decay = match pd >= epsilon {
                    true => pd,
                    false => epsilon.powi(2) / (-pd + 2.0 * epsilon),
                };

                total_deficit += pd_decay * f32::min(p_o.radius, p_i.radius);
            }
        }
    }
    assert!(NotNan::new(total_deficit).is_ok(), "total deficit is NaN");
    total_deficit
}

fn choose_inner_outer<'a>(
    sp1: &'a SPSurrogate,
    sp2: &'a SPSurrogate,
) -> (&'a SPSurrogate, &'a SPSurrogate) {
    //selects the surrogate with the smaller bounding circle as the inner surrogate
    let bp1 = &sp1.poles_bounding_circle;
    let bp2 = &sp2.poles_bounding_circle;
    match bp1.radius.partial_cmp(&bp2.radius).unwrap() {
        Ordering::Less => (sp1, sp2),
        Ordering::Greater => (sp2, sp1),
        Ordering::Equal => {
            //tiebreaker to ensure associativity
            match (bp1.center.0 + bp1.center.1)
                .partial_cmp(&(bp2.center.0 + bp2.center.1))
                .unwrap()
            {
                Ordering::Less => (sp1, sp2),
                _ => (sp2, sp1),
            }
        }
    }
}
