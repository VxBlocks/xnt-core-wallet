use triton_vm::prelude::*;

use crate::arithmetic::u32::is_u32::IsU32;
use crate::prelude::*;

/// Mimics [`u128::overflowing_add`].
///
/// ### Behavior
///
/// ```text
/// BEFORE: _ [rhs: u128] [lhs: u128]
/// AFTER:  _ [sum: u128] [is_overflow: bool]
/// ```
///
/// ### Preconditions
///
/// - all input arguments are properly [`BFieldCodec`] encoded
///
/// ### Postconditions
///
/// - the output is properly [`BFieldCodec`] encoded
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash)]
pub struct OverflowingAdd;

impl OverflowingAdd {
    /// Generate code to perform an addition on `u128`s.
    ///
    /// This function is called by both this snippet and
    /// [`SafeAdd`](super::safe_add::SafeAdd).
    ///
    /// ```text
    /// BEFORE: _ rhs_3 rhs_2 rhs_1 rhs_0 lhs_3 lhs_2 lhs_1 lhs_0
    /// AFTER:  _ sum_3 sum_2 sum_1 sum_0 is_overflow
    /// ```
    /// Don't forget to adapt the signature when using this function elsewhere.
    pub(crate) fn addition_code() -> Vec<LabelledInstruction> {
        triton_asm!(
            pick 4  // _ rhs_3 rhs_2 rhs_1 lhs_3 lhs_2 lhs_1 lhs_0 rhs_0
            add
            split   // _ rhs_3 rhs_2 rhs_1 lhs_3 lhs_2 lhs_1 (lhs_0 + rhs_0)_hi (lhs_0 + rhs_0)_lo
                    // _ rhs_3 rhs_2 rhs_1 lhs_3 lhs_2 lhs_1 carry_1            sum_0
            swap 5  // _ rhs_3 rhs_2 sum_0 lhs_3 lhs_2 lhs_1 carry_1 rhs_1
            add
            add
            split   // _ rhs_3 rhs_2 sum_0 lhs_3 lhs_2 carry_2 sum_1
            swap 5  // _ rhs_3 sum_1 sum_0 lhs_3 lhs_2 carry_2 rhs_2
            add
            add
            split   // _ rhs_3 sum_1 sum_0 lhs_3 carry_3 sum_2
            swap 5  // _ sum_2 sum_1 sum_0 lhs_3 carry_3 rhs_3
            add
            add
            split   // _ sum_2 sum_1 sum_0 carry_4 sum_3
            place 4 // _ sum_3 sum_2 sum_1 sum_0 carry_4
                    // _ sum_3 sum_2 sum_1 sum_0 is_overflow
        )
    }

    /// Assert that both `u128` operands consist of canonical `u32` limbs.
    ///
    /// The [`addition_code`](Self::addition_code) carry chain (and the
    /// subtraction in [`Sub`](super::sub::Sub)) is built from field `add`/`split`,
    /// which only equals integer arithmetic when every limb is `< 2^32`. A
    /// non-canonical limb (in `[2^32, p)`) would field-wrap mod `p`, silently
    /// under-reporting the result and suppressing the overflow/borrow assert. The
    /// add/sub family takes "limbs are valid u32s" as a precondition; this makes
    /// it self-enforced (matching [`i128::shift_right`][shr], which also asserts
    /// `is_u32` on its limbs) so a non-canonical, prover-decoded amount cannot
    /// reach the wrapping carry chain unchecked.
    ///
    /// [shr]: crate::arithmetic::i128::shift_right::ShiftRight
    ///
    /// ```text
    /// BEFORE: _ r_3 r_2 r_1 r_0 l_3 l_2 l_1 l_0
    /// AFTER:  _ r_3 r_2 r_1 r_0 l_3 l_2 l_1 l_0
    /// ```
    pub(crate) fn assert_operands_are_u32(library: &mut Library) -> Vec<LabelledInstruction> {
        let is_u32 = library.import(Box::new(IsU32));
        triton_asm! {
            // _ r_3 r_2 r_1 r_0 l_3 l_2 l_1 l_0
            dup 7 call {is_u32} assert error_id 610
            dup 6 call {is_u32} assert error_id 611
            dup 5 call {is_u32} assert error_id 612
            dup 4 call {is_u32} assert error_id 613
            dup 3 call {is_u32} assert error_id 614
            dup 2 call {is_u32} assert error_id 615
            dup 1 call {is_u32} assert error_id 616
            dup 0 call {is_u32} assert error_id 617
            // _ r_3 r_2 r_1 r_0 l_3 l_2 l_1 l_0
        }
    }
}

impl BasicSnippet for OverflowingAdd {
    fn parameters(&self) -> Vec<(DataType, String)> {
        ["lhs", "rhs"]
            .map(|s| (DataType::U128, s.to_owned()))
            .to_vec()
    }

    fn return_values(&self) -> Vec<(DataType, String)> {
        vec![
            (DataType::U128, "sum".to_owned()),
            (DataType::Bool, "overflow".to_owned()),
        ]
    }

    fn entrypoint(&self) -> String {
        "tasmlib_arithmetic_u128_overflowing_add".to_string()
    }

    fn code(&self, library: &mut Library) -> Vec<LabelledInstruction> {
        let assert_operands_are_u32 = Self::assert_operands_are_u32(library);
        triton_asm! {
            {self.entrypoint()}:
                {&assert_operands_are_u32}
                {&Self::addition_code()}
                return
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use rand::rngs::StdRng;

    use super::*;
    use crate::test_prelude::*;

    impl OverflowingAdd {
        fn assert_expected_add_behavior(&self, lhs: u128, rhs: u128) {
            let initial_stack = self.set_up_test_stack((rhs, lhs));

            let mut expected_stack = initial_stack.clone();
            self.rust_shadow(&mut expected_stack).unwrap();

            test_rust_equivalence_given_complete_state(
                &ShadowedClosure::new(Self),
                &initial_stack,
                &[],
                &NonDeterminism::default(),
                &None,
                Some(&expected_stack),
            );
        }

        pub fn edge_case_points() -> Vec<u128> {
            [0, 0x200000002fffffffffff908f8, 1 << 127, u128::MAX]
                .into_iter()
                .flat_map(|p| [p.checked_sub(1), Some(p), p.checked_add(1)])
                .flatten()
                .collect()
        }
    }

    impl Closure for OverflowingAdd {
        type Args = (u128, u128);

        fn rust_shadow(&self, stack: &mut Vec<BFieldElement>) -> Result<(), RustShadowError> {
            let (left, right) = pop_encodable::<Self::Args>(stack)?;
            let (sum, is_overflow) = left.overflowing_add(right);
            push_encodable(stack, &sum);
            push_encodable(stack, &is_overflow);
            Ok(())
        }

        fn pseudorandom_args(&self, seed: [u8; 32], _: Option<BenchmarkCase>) -> Self::Args {
            StdRng::from_seed(seed).random()
        }

        fn corner_case_args(&self) -> Vec<Self::Args> {
            let edge_case_points = Self::edge_case_points();

            edge_case_points
                .iter()
                .cartesian_product(&edge_case_points)
                .map(|(&l, &r)| (l, r))
                .collect()
        }
    }

    #[macro_rules_attr::apply(test)]
    fn rust_shadow() -> Result<(), RustShadowError> {
        ShadowedClosure::new(OverflowingAdd).test();
        Ok(())
    }

    #[macro_rules_attr::apply(test)]
    fn unit_test() {
        let snippet = OverflowingAdd;
        snippet.assert_expected_add_behavior(1u128 << 67, 1u128 << 67)
    }

    #[macro_rules_attr::apply(test)]
    fn overflow_test() {
        let test_overflowing_add = |a, b| {
            OverflowingAdd.assert_expected_add_behavior(a, b);
            OverflowingAdd.assert_expected_add_behavior(b, a);
        };

        test_overflowing_add(1, u128::MAX);
        test_overflowing_add(2, u128::MAX - 1);
        test_overflowing_add(1 << 127, 1 << 127);
        test_overflowing_add(u128::MAX, u128::MAX);

        for a in [31, 32, 33, 63, 64, 65, 95, 96, 97].map(|p| 1 << p) {
            test_overflowing_add(u128::MAX, a);
        }

        for i in 0..128 {
            let a = 1 << i;
            let b = u128::MAX - a + 1;
            debug_assert_eq!((0, true), a.overflowing_add(b), "i = {i}; a = {a}, b = {b}");

            test_overflowing_add(a, b);
        }
    }

    /// A non-canonical limb (in `[2^32, p)`) must be rejected before the carry
    /// chain can field-wrap it. Mirrors the audit PoC: `lhs_0 = p - 1`,
    /// `rhs_0 = 3` would otherwise wrap to `2` with the overflow flag cleared.
    #[macro_rules_attr::apply(test)]
    fn non_canonical_limb_is_rejected() {
        let mut stack = OverflowingAdd.set_up_test_stack((3, 0)); // (rhs, lhs)
        let top = stack.len() - 1;
        stack[top] = BFieldElement::new(BFieldElement::P - 1); // lhs_0 := p - 1
        test_assertion_failure(
            &ShadowedClosure::new(OverflowingAdd),
            InitVmState::with_stack(stack),
            &[617],
        );
    }
}

#[cfg(test)]
mod benches {
    use super::*;
    use crate::test_prelude::*;

    #[macro_rules_attr::apply(test)]
    fn benchmark() {
        ShadowedClosure::new(OverflowingAdd).bench();
    }
}
