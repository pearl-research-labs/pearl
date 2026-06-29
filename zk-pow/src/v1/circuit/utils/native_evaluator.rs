use plonky2::field::packed::PackedField;
use plonky2::field::types::Field;
use starky::constraint_consumer::ConstraintConsumer;

use super::evaluator::Evaluator;

/// An `Evaluator` for native field operations.
///
/// This struct implements the `Evaluator` trait for `PackedField`s, performing arithmetic
/// operations directly on the field elements and forwarding constraints to a
/// `ConstraintConsumer`.
pub(crate) struct NativeEvaluator<'a, P>
where
    P: PackedField,
{
    pub(crate) yield_constr: &'a mut ConstraintConsumer<P>,
}

impl<'a, P> NativeEvaluator<'a, P>
where
    P: PackedField,
{
    pub(crate) fn new(yield_constr: &'a mut ConstraintConsumer<P>) -> Self {
        Self { yield_constr }
    }
}

impl<'a, P> Evaluator<P, P::Scalar> for NativeEvaluator<'a, P>
where
    P: PackedField,
    P::Scalar: Field,
    P: From<P::Scalar>,
{
    fn add(&mut self, a: P, b: P) -> P {
        a + b
    }

    fn sub(&mut self, a: P, b: P) -> P {
        a - b
    }

    fn mul(&mut self, a: P, b: P) -> P {
        a * b
    }

    fn i32(&mut self, s: i32) -> P {
        if s >= 0 {
            self.scalar(P::Scalar::from_canonical_u32(s as u32))
        } else {
            self.scalar(-P::Scalar::from_canonical_u32(-s as u32))
        }
    }

    fn u64(&mut self, s: u64) -> P {
        self.scalar(P::Scalar::from_canonical_u64(s))
    }

    fn scalar(&mut self, s: P::Scalar) -> P {
        P::from(s)
    }

    fn constraint(&mut self, constraint: P) {
        self.yield_constr.constraint(constraint);
    }

    fn constraint_transition(&mut self, constraint: P) {
        self.yield_constr.constraint_transition(constraint);
    }

    fn constraint_first_row(&mut self, constraint: P) {
        self.yield_constr.constraint_first_row(constraint);
    }

    fn constraint_last_row(&mut self, constraint: P) {
        self.yield_constr.constraint_last_row(constraint);
    }
}
