/// A trait for evaluating STARK constraints.
///
/// This trait abstracts over two modes of operation:
/// 1. Native evaluation, where computations are performed on `PackedField`s.
/// 2. Circuit evaluation, where computations are performed on `ExtensionTarget`s within a
///    `CircuitBuilder`.
///
/// By using this trait, we can write a single constraint evaluation function that is generic
/// over the evaluation mode.
///
/// `V`: The value type, e.g., `PackedField` or `ExtensionTarget`. `V` must be copyable,
/// and support basic arithmetic operations.
/// `S`: The scalar type, used for public inputs. For native evaluation, this is a field extension.
/// For circuit evaluation, this is the same as `V`.
pub(crate) trait Evaluator<V, S>
where
    V: Copy, // Value type, in which we perform arithmetic operations
    S: Copy,
{
    /// Adds two values.
    fn add(&mut self, a: V, b: V) -> V;

    /// Subtracts the second value from the first.
    fn sub(&mut self, a: V, b: V) -> V;

    /// Multiplies two values.
    fn mul(&mut self, a: V, b: V) -> V;

    /// Converts u32 to a value.
    fn i32(&mut self, s: i32) -> V;

    fn u64(&mut self, s: u64) -> V;

    /// Converts a scalar to a value.
    fn scalar(&mut self, s: S) -> V;

    /// Adds a constraint to every row.
    fn constraint(&mut self, constraint: V);

    /// Adds a transition constraint (using current and next row).
    fn constraint_transition(&mut self, constraint: V);

    /// Adds a constraint that only applies to the first row.
    fn constraint_first_row(&mut self, constraint: V);

    /// Adds a constraint that only applies to the last row.
    fn constraint_last_row(&mut self, constraint: V);

    // syntactic-sugar function / specialized impls

    /// multiply-add: a*b + c
    fn mad(&mut self, a: V, b: V, c: V) -> V {
        let ab = self.mul(a, b);
        self.add(ab, c)
    }

    /// multiply-subtract: a*b - c
    fn msub(&mut self, a: V, b: V, c: V) -> V {
        let ab = self.mul(a, b);
        self.sub(ab, c)
    }

    fn polyval(&mut self, coeffs: &[V], x: V) -> V {
        if coeffs.is_empty() {
            return self.i32(0);
        }
        let mut res = *coeffs.last().unwrap();
        for &c in coeffs.iter().rev().skip(1) {
            res = self.mad(res, x, c);
        }
        res
    }

    fn constraint_bool(&mut self, a: V) {
        let a2ma = self.msub(a, a, a);
        self.constraint(a2ma);
    }

    /// Adds a constraint that a == b.
    fn constraint_eq(&mut self, a: V, b: V) {
        let sub = self.sub(a, b);
        self.constraint(sub);
    }

    fn constraint_eq_if(&mut self, condition: V, a: V, b: V) {
        let diff = self.sub(a, b);
        let cond_diff = self.mul(condition, diff);
        self.constraint(cond_diff);
    }

    /// Adds a transition constraint that a == b.
    fn constraint_transition_eq(&mut self, a: V, b: V) {
        let sub = self.sub(a, b);
        self.constraint_transition(sub);
    }

    fn constraint_first_row_eq(&mut self, a: V, b: V) {
        let sub = self.sub(a, b);
        self.constraint_first_row(sub);
    }

    fn constraint_last_row_eq(&mut self, a: V, b: V) {
        let sub = self.sub(a, b);
        self.constraint_last_row(sub);
    }

    fn xor_bit(&mut self, a: V, b: V) -> V {
        let ab = self.mul(a, b);
        let double_ab = self.add(ab, ab);
        let apb = self.add(a, b);
        self.sub(apb, double_ab)
    }

    fn mux(&mut self, choice: V, option0: V, option1: V) -> V {
        // 0 for a, 1 for b
        let diff = self.sub(option1, option0);
        self.mad(choice, diff, option0)
    }

    /// assumes a.len() == b.len()
    fn inner_product(&mut self, a: &[V], b: &[V]) -> V {
        debug_assert!(a.len() == b.len());
        let mut res = self.mul(a[0], b[0]);
        for i in 1..a.len() {
            res = self.mad(a[i], b[i], res);
        }
        res
    }

    /// assumes a.len() >= 2
    fn sum(&mut self, a: &[V]) -> V {
        debug_assert!(a.len() >= 2);
        let mut res = self.add(a[0], a[1]);
        for &v in a.iter().skip(2) {
            res = self.add(res, v);
        }
        res
    }
}
