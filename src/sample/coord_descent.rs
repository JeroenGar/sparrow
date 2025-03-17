use crate::config::{CD_STEP_FAIL, CD_STEP_INIT_RATIO, CD_STEP_LIMIT_RATIO, CD_STEP_SUCCESS};
use crate::eval::sample_eval::{SampleEval, SampleEvaluator};
use jagua_rs::geometry::d_transformation::DTransformation;
use jagua_rs::geometry::primitives::point::Point;
use log::trace;
use rand::Rng;
use std::cmp::Ordering;
use std::fmt::Debug;

pub fn coordinate_descent(
    (init_dt, init_eval): (DTransformation, SampleEval),
    evaluator: &mut impl SampleEvaluator,
    min_dim: f32,
    rng: &mut impl Rng,
) -> (DTransformation, SampleEval) {
    let mut counter = 0;
    let init_pos = init_dt.translation().into();
    let rot = init_dt.rotation.into();

    let mut cd_state = CDState {
        pos: init_pos,
        eval: init_eval,
        axis: AXES[rng.random_range(0..4)],
        steps: (min_dim * CD_STEP_INIT_RATIO, min_dim * CD_STEP_INIT_RATIO),
        step_limit: min_dim * CD_STEP_LIMIT_RATIO,
    };

    while let Some([c0, c1]) = cd_state.gen_candidates() {
        let c0_eval = evaluator.eval(DTransformation::new(rot, c0.into()), Some(cd_state.eval));
        let c1_eval = evaluator.eval(DTransformation::new(rot, c1.into()), Some(cd_state.eval));

        counter += 2;

        let min_state = [(c0, c0_eval), (c1, c1_eval)]
            .into_iter()
            .min_by_key(|(_, e)| *e)
            .unwrap();

        cd_state.evolve(min_state, rng);
        trace!("CD: {:?}", cd_state);

        debug_assert!(counter < 10_000);
    }
    trace!(
        "CD: {} evals, t: ({:.3}, {:.3}) -> ({:.3}, {:.3}), eval: {:?}",
        counter, init_pos.0, init_pos.1, cd_state.pos.0, cd_state.pos.1, cd_state.eval
    );
    (DTransformation::new(rot, cd_state.pos.into()), cd_state.eval)
}

#[derive(Debug)]
struct CDState {
    pub pos: Point,
    pub eval: SampleEval,
    pub axis: CDAxis,
    pub steps: (f32, f32),
    pub step_limit: f32,
}

impl CDState {
    pub fn evolve(&mut self, evolved_state: (Point, SampleEval), rng: &mut impl Rng) {
        match evolved_state.1.cmp(&self.eval){
            Ordering::Less => {
                (self.pos, self.eval) = evolved_state;
                self.adjust_steps_and_axis(true, rng);
            },
            Ordering::Equal => {
                (self.pos, self.eval) = evolved_state;
                self.adjust_steps_and_axis(false, rng);
            },
            Ordering::Greater => {
                self.adjust_steps_and_axis(false, rng);
            },
        }
    }

    fn adjust_steps_and_axis(&mut self, improved: bool, rng: &mut impl Rng) {
        let m = if improved { CD_STEP_SUCCESS } else { CD_STEP_FAIL };
        let (sx, sy) = self.steps;

        self.steps = match self.axis {
            CDAxis::Horizontal => (sx * m, sy),
            CDAxis::Vertical => (sx, sy * m),
            //since both axis are involved, adjust both steps but less severely
            CDAxis::ForwardDiag | CDAxis::BackwardDiag => (sx * m.sqrt(), sy * m.sqrt()),
        };
        self.axis.cycle(rng);
    }

    pub fn gen_candidates(&self) -> Option<[Point; 2]> {
        let p = self.pos;
        let (sx, sy) = self.steps;

        if sx < self.step_limit && sy < self.step_limit {
            // Stop generating candidates if both steps have reached the limit
            None
        } else {
            let c = match self.axis {
                CDAxis::Horizontal => [Point(p.0 + sx, p.1), Point(p.0 - sx, p.1)],
                CDAxis::Vertical => [Point(p.0, p.1 + sy), Point(p.0, p.1 - sy)],
                CDAxis::ForwardDiag => [Point(p.0 + sx, p.1 + sy), Point(p.0 - sx, p.1 - sy)],
                CDAxis::BackwardDiag => [Point(p.0 - sx, p.1 + sy), Point(p.0 + sx, p.1 - sy)],
            };
            Some(c)
        }
    }
}

const AXES: [CDAxis; 4] = [
    CDAxis::Horizontal,
    CDAxis::Vertical,
    CDAxis::ForwardDiag,
    CDAxis::BackwardDiag,
];

#[derive(Clone, Debug, Copy)]
enum CDAxis {
    ///left and right
    Horizontal,
    ///up and down
    Vertical,
    ///up-right and down-left
    ForwardDiag,
    ///up-left and down-right
    BackwardDiag,
}

impl CDAxis {
    fn cycle(&mut self, rng: &mut impl Rng) {
        // *self = match self {
        //     CDAxis::Horizontal => CDAxis::Vertical,
        //     CDAxis::Vertical => CDAxis::ForwardDiag,
        //     CDAxis::ForwardDiag => CDAxis::BackwardDiag,
        //     CDAxis::BackwardDiag => CDAxis::Horizontal,
        // }
        *self = match rng.random_range(0..4) {
            0 => CDAxis::Horizontal,
            1 => CDAxis::Vertical,
            2 => CDAxis::ForwardDiag,
            _ => CDAxis::BackwardDiag,
        };
    }
}
