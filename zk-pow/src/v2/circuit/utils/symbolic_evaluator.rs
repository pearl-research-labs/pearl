use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use starky::constraint_consumer::RecursiveConstraintConsumer;

use super::evaluator::Evaluator;

/// An `Evaluator` for symbolic circuit operations.
///
/// This struct implements the `Evaluator` trait for `ExtensionTarget`s, performing arithmetic
/// operations using a `CircuitBuilder` and forwarding constraints to a
/// `RecursiveConstraintConsumer`.
pub(crate) struct SymbolicEvaluator<'a, F: RichField + Extendable<D>, const D: usize> {
    pub(crate) builder: &'a mut CircuitBuilder<F, D>,
    pub(crate) yield_constr: &'a mut RecursiveConstraintConsumer<F, D>,
}

impl<'a, F: RichField + Extendable<D>, const D: usize> SymbolicEvaluator<'a, F, D> {
    pub(crate) fn new(builder: &'a mut CircuitBuilder<F, D>, yield_constr: &'a mut RecursiveConstraintConsumer<F, D>) -> Self {
        Self { builder, yield_constr }
    }
}

impl<'a, F, const D: usize> Evaluator<ExtensionTarget<D>, ExtensionTarget<D>> for SymbolicEvaluator<'a, F, D>
where
    F: Field + RichField + Extendable<D>,
{
    fn add(&mut self, a: ExtensionTarget<D>, b: ExtensionTarget<D>) -> ExtensionTarget<D> {
        self.builder.add_extension(a, b)
    }

    fn sub(&mut self, a: ExtensionTarget<D>, b: ExtensionTarget<D>) -> ExtensionTarget<D> {
        self.builder.sub_extension(a, b)
    }

    fn mul(&mut self, a: ExtensionTarget<D>, b: ExtensionTarget<D>) -> ExtensionTarget<D> {
        self.builder.mul_extension(a, b)
    }

    fn mad(&mut self, a: ExtensionTarget<D>, b: ExtensionTarget<D>, c: ExtensionTarget<D>) -> ExtensionTarget<D> {
        self.builder.mul_add_extension(a, b, c)
    }

    fn msub(&mut self, a: ExtensionTarget<D>, b: ExtensionTarget<D>, c: ExtensionTarget<D>) -> ExtensionTarget<D> {
        self.builder.mul_sub_extension(a, b, c)
    }

    fn i32(&mut self, s: i32) -> ExtensionTarget<D> {
        if s == 0 {
            self.builder.zero_extension()
        } else if s == 1 {
            self.builder.one_extension()
        } else if s == 2 {
            self.builder.two_extension()
        } else if s == -1 {
            self.builder.neg_one_extension()
        } else if s >= 0 {
            self.builder.constant_extension(F::Extension::from_canonical_u32(s as u32))
        } else {
            self.builder.constant_extension(-F::Extension::from_canonical_u32(-s as u32)) // s < 0
        }
    }

    fn u64(&mut self, s: u64) -> ExtensionTarget<D> {
        if s < 3 {
            self.i32(s as i32)
        } else {
            self.builder.constant_extension(F::Extension::from_canonical_u64(s))
        }
    }

    fn scalar(&mut self, s: ExtensionTarget<D>) -> ExtensionTarget<D> {
        s
    }

    fn constraint(&mut self, constraint: ExtensionTarget<D>) {
        self.yield_constr.constraint(self.builder, constraint);
    }

    fn constraint_transition(&mut self, constraint: ExtensionTarget<D>) {
        self.yield_constr.constraint_transition(self.builder, constraint);
    }

    fn constraint_first_row(&mut self, constraint: ExtensionTarget<D>) {
        self.yield_constr.constraint_first_row(self.builder, constraint);
    }

    fn constraint_last_row(&mut self, constraint: ExtensionTarget<D>) {
        self.yield_constr.constraint_last_row(self.builder, constraint);
    }
}
