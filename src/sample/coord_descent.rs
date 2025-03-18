use crate::config::{CD_STEP_FAIL, CD_STEP_INIT_RATIO, CD_STEP_LIMIT_RATIO, CD_STEP_SUCCESS};
use crate::eval::sample_eval::{SampleEval, SampleEvaluator};
use jagua_rs::geometry::d_transformation::DTransformation;
use jagua_rs::geometry::primitives::point::Point;
use log::trace;
use rand::Rng;
use std::cmp::Ordering;
use std::fmt::Debug;

/// todo: write docs for this function
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
        axis: CDAxis::random(rng),
        steps: (min_dim * CD_STEP_INIT_RATIO, min_dim * CD_STEP_INIT_RATIO),
        step_limit: min_dim * CD_STEP_LIMIT_RATIO,
    };

    while let Some([c0, c1]) = cd_state.ask() {
        let c0_eval = evaluator.eval(DTransformation::new(rot, c0.into()), Some(cd_state.eval));
        let c1_eval = evaluator.eval(DTransformation::new(rot, c1.into()), Some(cd_state.eval));

        counter += 2;

        let min_state = [(c0, c0_eval), (c1, c1_eval)]
            .into_iter()
            .min_by_key(|(_, e)| *e)
            .unwrap();

        cd_state.tell(min_state, rng);
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
    pub fn tell(&mut self, (pos, eval): (Point, SampleEval), rng: &mut impl Rng) {
        let eval_cmp = eval.cmp(&self.eval);

        if eval_cmp != Ordering::Greater {
            // Update the state if not worse
            (self.pos, self.eval) = (pos, eval);
        }

        // Multiply step size of active axis
        let m = match eval_cmp {
            Ordering::Less => CD_STEP_SUCCESS,
            _ => CD_STEP_FAIL,
        };

        match self.axis {
            CDAxis::Horizontal => self.steps.0 *= m,
            CDAxis::Vertical => self.steps.1 *= m,
            CDAxis::ForwardDiag | CDAxis::BackwardDiag => {
                //since both axis are involved, adjust both steps but less severely
                self.steps.0 *= m.sqrt();
                self.steps.1 *= m.sqrt();
            }
        };

        // Change axis if not improved
        if eval_cmp != Ordering::Less {
            self.axis = CDAxis::random(rng);
        }
    }

    pub fn ask(&self) -> Option<[Point; 2]> {
        let (sx, sy) = self.steps;

        if sx < self.step_limit && sy < self.step_limit {
            // Stop generating candidates if both steps have reached the limit
            None
        } else {
            // Generate two candidates on either side of the current position
            let p = self.pos;
            let c = match self.axis {
                CDAxis::Horizontal => [Point(p.0 + sx, p.1), Point(p.0 - sx, p.1)],
                CDAxis::Vertical => [Point(p.0, p.1 + sy), Point(p.0, p.1 - sy)],
                CDAxis::ForwardDiag => [Point(p.0 + sx.sqrt(), p.1 + sy.sqrt()), Point(p.0 - sx.sqrt(), p.1 - sy.sqrt())],
                CDAxis::BackwardDiag => [Point(p.0 - sx.sqrt(), p.1 + sy.sqrt()), Point(p.0 + sx.sqrt(), p.1 - sy.sqrt())],
            };
            Some(c)
        }
    }
}

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
    fn random(rng: &mut impl Rng) -> Self {
        match rng.random_range(0..4) {
            0 => CDAxis::Horizontal,
            1 => CDAxis::Vertical,
            2 => CDAxis::ForwardDiag,
            3 => CDAxis::BackwardDiag,
            _ => unreachable!(),
        }
    }
}