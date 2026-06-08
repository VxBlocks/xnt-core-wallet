use std::fmt::Debug;
use std::marker::PhantomData;

use triton_vm::prelude::*;

use crate::prelude::*;

/// Verify size-indicator integrity of preloaded data, return size.
///
/// This snippet has a deliberately narrow responsibility. It certifies that the
/// embedded *size indicators* of a `BFieldCodec`-encoded `T` in non-deterministic
/// memory are mutually consistent — every field and element has the size it claims
/// — so that subsequent `get_field`/`destructure` accesses land on the correct
/// offsets. It additionally checks that the entire structure lies within the
/// non-deterministic region (a valid `u32` address range).
///
/// It deliberately does **not** validate *leaf-value canonicality*. A value whose
/// length is fixed by its type — e.g. a `u32`, a `bool`, or the coefficients of a
/// `Polynomial` — is accepted regardless of its contents: a `u32` word may hold a
/// value `>= 2^32`, a `bool` word a value other than `0`/`1`, and so on. Such a
/// word still occupies the correct number of memory cells, which is all that
/// "size-indicator integrity" concerns. This is therefore strictly *weaker* than
/// `BFieldCodec::decode`, which also range-checks leaves; **passing this gate does
/// not imply the data is a canonical encoding of `T`.**
///
/// A caller that consumes a leaf as an in-range value is responsible for enforcing
/// that range itself. Most consumers do so for free — the u32/arithmetic
/// instructions (`lt`, `split`, `pop_count`, …) crash on non-canonical words — but
/// a raw word used as a hash preimage, an equality operand, or a `skiz`/`assert`
/// discriminant does not self-defend. Treating "`VerifyNdSiIntegrity` passed" as
/// "the witness is canonical" is a mistake.
///
/// Crashes the VM if the structure in question is not entirely contained within
/// the non-deterministic section of memory as defined in the memory layout.
#[derive(Clone, Debug)]
pub struct VerifyNdSiIntegrity<PreloadedData: TasmObject + Clone + Debug> {
    _phantom_data: PhantomData<PreloadedData>,
}

impl<T: TasmObject + Clone + Debug> Default for VerifyNdSiIntegrity<T> {
    fn default() -> Self {
        Self {
            _phantom_data: PhantomData,
        }
    }
}

impl<T: TasmObject + Clone + Debug> BasicSnippet for VerifyNdSiIntegrity<T> {
    fn parameters(&self) -> Vec<(DataType, String)> {
        vec![(DataType::VoidPointer, "*struct".to_owned())]
    }

    fn return_values(&self) -> Vec<(DataType, String)> {
        vec![(DataType::U32, "struct size".to_owned())]
    }

    fn entrypoint(&self) -> String {
        let name = T::label_friendly_name();
        format!("tasmlib_structure_verify_nd_si_integrity___{name}")
    }

    fn code(&self, library: &mut Library) -> Vec<LabelledInstruction> {
        let entrypoint = self.entrypoint();
        let si_integrity_check_code = T::compute_size_and_assert_valid_size_indicator(library);

        triton_asm!(
            {entrypoint}:
                // _ *struct

                dup 0
                {&si_integrity_check_code}
                // _ *struct calculated_size

                /* Verify that both pointer and end of struct is in ND-region */
                dup 1
                // _ *struct calculated_size *struct

                pop_count // verifies that `*struct` is a valid u32
                pop 1
                // _ *struct calculated_size

                swap 1
                dup 1
                add
                // _ calculated_size *end_of_struct

                pop_count // verifies that end of struct is located in ND-region
                pop 1
                // _ calculated_size

                return
        )
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Debug;

    use arbitrary::Arbitrary;
    use arbitrary::Unstructured;
    use num_traits::ConstZero;
    use twenty_first::util_types::mmr::mmr_successor_proof::MmrSuccessorProof;

    use super::*;
    use crate::memory::encode_to_memory;
    use crate::neptune::neptune_like_types_for_tests::*;
    use crate::test_prelude::*;

    impl<T> VerifyNdSiIntegrity<T>
    where
        T: TasmObject + BFieldCodec + for<'a> Arbitrary<'a> + Debug + Clone,
    {
        fn initial_state(&self, address: BFieldElement, t: T) -> AccessorInitialState {
            let mut memory = HashMap::default();
            encode_to_memory(&mut memory, address, &t);

            AccessorInitialState {
                stack: [self.init_stack_for_isolated_run(), vec![address]].concat(),
                memory,
            }
        }

        fn prepare_random_object(&self, randomness: &[u8]) -> T {
            let unstructured = Unstructured::new(randomness);
            T::arbitrary_take_rest(unstructured).unwrap()
        }
    }

    impl<T> Accessor for VerifyNdSiIntegrity<T>
    where
        T: TasmObject + for<'a> Arbitrary<'a> + Debug + Clone,
    {
        fn rust_shadow(
            &self,
            stack: &mut Vec<BFieldElement>,
            memory: &HashMap<BFieldElement, BFieldElement>,
        ) -> Result<(), RustShadowError> {
            // A successful decode implies valid size indicators, so `decode` is
            // used here as a convenient proxy for the property this snippet
            // checks. Note it is *stricter* than the snippet (it also range-checks
            // leaf values; see the struct docs), so it would reject some
            // non-canonical inputs the TASM accepts. The test harness only ever
            // generates canonical objects (via `Arbitrary`), so the two never
            // diverge in practice — do not "fix" the snippet to match this.
            let pointer = stack.pop().ok_or(RustShadowError::StackUnderflow)?;
            let obj = T::decode_from_memory(memory, pointer)
                .map_err(|_| RustShadowError::DecodingError)?;
            let encoding_len = obj.encode().len();
            let encoding_len: u32 = encoding_len
                .try_into()
                .map_err(|_| RustShadowError::UsizeToU32Error)?;

            // Verify contained in ND-region
            let start_address: u32 = pointer
                .value()
                .try_into()
                .map_err(|_| RustShadowError::U64ToU32Error)?;
            let _end_address = start_address
                .checked_add(encoding_len)
                .ok_or(RustShadowError::ArithmeticOverflow)?;

            stack.push(bfe!(u64::from(encoding_len)));

            Ok(())
        }

        fn pseudorandom_initial_state(
            &self,
            seed: [u8; 32],
            _bench_case: Option<BenchmarkCase>,
        ) -> AccessorInitialState {
            let mut rng = StdRng::from_seed(seed);

            let t: T = {
                let mut randomness = [0u8; 100000];
                rng.fill(&mut randomness);
                self.prepare_random_object(&randomness)
            };

            let address: u32 = rng.random_range(0..(1 << 30));
            let address = bfe!(address);
            self.initial_state(address, t)
        }

        fn corner_case_initial_states(&self) -> Vec<AccessorInitialState> {
            // This *should* always return `None` if `T: Option<S>`, and empty
            // vec if type is Vec<T>. So some notion of "empty" or default.
            let empty_struct: T = {
                let unstructured = Unstructured::new(&[]);
                T::arbitrary_take_rest(unstructured).unwrap()
            };

            println!("empty_struct:\n{empty_struct:?}");
            let empty_struct_at_zero = self.initial_state(BFieldElement::ZERO, empty_struct);

            vec![empty_struct_at_zero]
        }
    }

    macro_rules! test_case {
        (fn $test_name:ident for $t:ty) => {
            #[macro_rules_attr::apply(test)]
            fn $test_name() {
                ShadowedAccessor::new(VerifyNdSiIntegrity::<$t>::default()).test();
            }
        };
        (fn $test_name:ident for new type $t:ty: $($type_declaration:tt)*) => {
            #[macro_rules_attr::apply(test)]
            fn $test_name() {
                #[derive(Debug, Clone, TasmObject, BFieldCodec, Arbitrary)]
                $($type_declaration)*
                ShadowedAccessor::new(VerifyNdSiIntegrity::<$t>::default()).test();
            }
        };
    }

    mod simple_struct {
        use super::*;
        use crate::test_helpers::negative_test;
        use crate::test_helpers::test_assertion_failure;

        #[derive(Debug, Clone, TasmObject, BFieldCodec, Arbitrary)]
        struct TestStruct {
            a: Vec<u128>,
            b: Digest,
            c: Vec<Digest>,
        }

        test_case! { fn test_pbt_simple_struct for TestStruct }

        #[macro_rules_attr::apply(test)]
        fn struct_not_contained_in_nd_region() {
            let snippet = VerifyNdSiIntegrity::<TestStruct>::default();

            let t = snippet.prepare_random_object(&[]);
            let begin_address = bfe!((1u64 << 32) - 4);
            let init_state = snippet.initial_state(begin_address, t.clone());

            let actual_size = t.encode().len();
            let end_address = begin_address + bfe!(actual_size as u64);
            let expected_err =
                InstructionError::OpStackError(OpStackError::FailedU32Conversion(end_address));
            negative_test(
                &ShadowedAccessor::new(snippet),
                init_state.into(),
                &[expected_err],
            )
        }

        #[macro_rules_attr::apply(test)]
        fn struct_does_not_start_in_nd_region() {
            let snippet = VerifyNdSiIntegrity::<TestStruct>::default();

            let begin_address = bfe!(-4);
            let init_state =
                snippet.initial_state(begin_address, snippet.prepare_random_object(&[]));
            let expected_err =
                InstructionError::OpStackError(OpStackError::FailedU32Conversion(begin_address));
            negative_test(
                &ShadowedAccessor::new(snippet),
                init_state.into(),
                &[expected_err],
            )
        }

        #[macro_rules_attr::apply(test)]
        fn lie_about_digest_vec_size() {
            let snippet = VerifyNdSiIntegrity::<TestStruct>::default();

            let begin_address = bfe!(4);
            let mut init_state =
                snippet.initial_state(begin_address, snippet.prepare_random_object(&[]));
            let true_value = init_state.memory[&begin_address];
            init_state
                .memory
                .insert(begin_address, true_value + bfe!(1));

            test_assertion_failure(&ShadowedAccessor::new(snippet), init_state.into(), &[181])
        }

        #[macro_rules_attr::apply(test)]
        fn lie_about_digest_vec_len() {
            let snippet = VerifyNdSiIntegrity::<TestStruct>::default();

            let begin_address = bfe!(4);
            let mut init_state =
                snippet.initial_state(begin_address, snippet.prepare_random_object(&[42u8; 20000]));
            let vec_digest_len_indicator = begin_address + bfe!(1);
            let true_value = init_state.memory[&vec_digest_len_indicator];
            init_state
                .memory
                .insert(vec_digest_len_indicator, true_value + bfe!(1));

            test_assertion_failure(&ShadowedAccessor::new(snippet), init_state.into(), &[181])
        }
    }

    mod option_types {
        use super::*;
        use crate::test_helpers::test_assertion_failure;

        #[derive(Debug, Clone, TasmObject, BFieldCodec, Arbitrary)]
        struct StatSizedPayload {
            a: Option<Digest>,
        }

        #[derive(Debug, Clone, TasmObject, BFieldCodec, Arbitrary, Default)]
        struct DynSizedPayload {
            a: Option<Vec<u128>>,
            b: Digest,
            c: Vec<Vec<BFieldElement>>,
            d: Option<Vec<Option<BFieldElement>>>,
        }

        test_case! {fn test_option_stat_sized_elem for StatSizedPayload }
        test_case! {fn test_option_dyn_sized_elem for DynSizedPayload }

        #[macro_rules_attr::apply(test)]
        fn lie_about_option_payload_field_size() {
            let snippet = VerifyNdSiIntegrity::<DynSizedPayload>::default();

            let begin_address = bfe!(4);
            let randomness = rand::random::<[u8; 100_000]>();
            let obj = snippet.prepare_random_object(&randomness);
            let true_init_state = snippet.initial_state(begin_address, obj.clone());

            /*  Lie about size of field 'd'*/
            let mut manipulated_si_outer = true_init_state.clone();
            let outer_option_payload_si_ptr = begin_address; // field size-indicator of `d` field
            let true_value = true_init_state.memory[&outer_option_payload_si_ptr];
            manipulated_si_outer
                .memory
                .insert(outer_option_payload_si_ptr, true_value + bfe!(1));

            test_assertion_failure(
                &ShadowedAccessor::new(snippet),
                manipulated_si_outer.into(),
                &[181],
            );
        }

        #[macro_rules_attr::apply(test)]
        fn illegal_discriminant_value_for_option() {
            let snippet = VerifyNdSiIntegrity::<DynSizedPayload>::default();

            let obj = DynSizedPayload::default();
            let begin_address = bfe!(4);
            let mut manipulated_init_state = snippet.initial_state(begin_address, obj.clone());
            let option_discriminant_ptr = begin_address + bfe!(1);
            manipulated_init_state
                .memory
                .insert(option_discriminant_ptr, bfe!(2));

            test_assertion_failure(
                &ShadowedAccessor::new(snippet.clone()),
                manipulated_init_state.into(),
                &[200],
            );
        }

        #[macro_rules_attr::apply(test)]
        fn lie_about_option_payload_size() {
            let snippet = VerifyNdSiIntegrity::<DynSizedPayload>::default();

            let obj = DynSizedPayload {
                d: Some(vec![Some(bfe!(14)), None, Some(bfe!(15))]),
                ..Default::default()
            };
            let begin_address = bfe!(4);
            let true_init_state = snippet.initial_state(begin_address, obj.clone());

            /*  Lie about size of payload of outer Some(...)*/
            let mut add_one = true_init_state.clone();
            let len_of_dyn_sized_list_elem_0 = begin_address + bfe!(3);
            let true_value = true_init_state.memory[&len_of_dyn_sized_list_elem_0];
            add_one
                .memory
                .insert(len_of_dyn_sized_list_elem_0, true_value + bfe!(1));

            test_assertion_failure(
                &ShadowedAccessor::new(snippet.clone()),
                add_one.into(),
                &[211],
            );

            let mut sub_one = true_init_state.clone();
            sub_one
                .memory
                .insert(len_of_dyn_sized_list_elem_0, true_value - bfe!(1));

            test_assertion_failure(&ShadowedAccessor::new(snippet), sub_one.into(), &[211]);
        }
    }

    test_case! { fn test_stat_sized_tuple for new type TestStruct:
        struct TestStruct { field: (Digest, XFieldElement) }
    }

    test_case! { fn test_dyn_state_sized_tuple_right_dyn for new type RightIsDyn:
        struct RightIsDyn { field: (Digest, Vec<XFieldElement>) }
    }

    test_case! { fn test_dyn_state_sized_tuple_left_dyn for new type LeftIsDyn:
        struct LeftIsDyn { field: (Vec<XFieldElement>, Digest) }
    }

    test_case! { fn test_dyn_state_sized_tuple_both_dyn for new type BothDyn:
        struct BothDyn { field: (Vec<XFieldElement>, Vec<XFieldElement>) }
    }

    test_case! { fn proof for Proof }
    test_case! { fn coin for CoinLookalike }
    test_case! { fn utxo for UtxoLookalike }
    test_case! { fn salted_utxos for SaltedUtxosLookalike }
    test_case! { fn collect_lock_scripts_witness for CollectLockScriptsWitnessLookalike }
    test_case! { fn collect_type_scripts_witness for CollectTypeScriptsWitnessLookalike }
    test_case! { fn claim for Claim }
    test_case! { fn neptune_coins for NeptuneCoinsLookalike }
    test_case! { fn option_neptune_coins for Option<NeptuneCoinsLookalike> }
    test_case! { fn chunk for ChunkLookalike }
    test_case! { fn chunk_dictionary for ChunkDictionaryLookalike }
    test_case! { fn proof_collection for ProofCollectionLookalike }
    test_case! { fn absolute_index_set for AbsoluteIndexSetLookalike }
    test_case! { fn removal_record for RemovalRecordLookalike }
    test_case! { fn addition_record for AdditionRecordLookalike }
    test_case! { fn public_announcement for PublicAnnouncementLookalike }
    test_case! { fn timestamp for TimestampLookalike }
    test_case! { fn transaction_kernel for TransactionKernelLookalike }
    test_case! { fn kernel_to_outputs_witness for KernelToOutputsWitnessLookalike }
    test_case! { fn merge_witness for MergeWitnessLookalike }
    test_case! { fn ms_membership_proof for MsMembershipProofLookalike }
    test_case! { fn removal_records_witness for RemovalRecordsIntegrityWitnessLookalike }
    test_case! { fn update_witness for UpdateWitnessLookalike }
    test_case! { fn lock_script_and_witness for LockScriptAndWitnessLookalike }
    test_case! { fn mutator_set_accumulator for MutatorSetAccumulatorLookalike }
    test_case! { fn primitive_witness for PrimitiveWitnessLookalike }
    test_case! { fn mmr_successor_proof for MmrSuccessorProof }

    mod modular_wrap_on_static_elem_vec_len {
        use super::*;
        use crate::execute_with_terminal_state;
        use crate::structure::tasm_object::DEFAULT_MAX_DYN_FIELD_SIZE;

        #[derive(Debug, Clone, TasmObject, BFieldCodec)]
        struct OnlyDigestVec {
            #[allow(dead_code)]
            digests: Vec<Digest>,
        }

        /// Regression test for the audit finding: `VerifyNdSiIntegrity` used to
        /// accept a static-element `Vec` whose length word is ~2^64 while only a
        /// tiny region was validated. The length is multiplied by the (static)
        /// element size with the field `mul`; without bounding it a prover could
        /// pick `list_len = (field_size - 1) * elem_size^{-1} mod p` to satisfy
        /// both id-180 (`field_size < MAX_OFFSET`) and id-181
        /// (`calculated == field_size`). The fix bounds `list_len < MAX_OFFSET`
        /// before the multiply (`error_id 212`), making the wrap impossible.
        #[macro_rules_attr::apply(test)]
        fn si_integrity_rejects_overlong_static_elem_vec() {
            let snippet = VerifyNdSiIntegrity::<OnlyDigestVec>::default();
            let program = Program::new(&snippet.link_for_isolated_run());

            let struct_ptr = bfe!(4);
            let elem_size = bfe!(Digest::LEN as u64); // 5

            let run = |field_si: BFieldElement, list_len: BFieldElement| {
                let mut ram = HashMap::new();
                ram.insert(struct_ptr, field_si); // size indicator of `digests`
                ram.insert(struct_ptr + bfe!(1), list_len); // Vec length word
                let nd = NonDeterminism::default().with_ram(ram);
                let stack = [snippet.init_stack_for_isolated_run(), vec![struct_ptr]].concat();
                execute_with_terminal_state(program.clone(), &[], &stack, &nd, None)
            };

            // The binding-break primitive: a wrapped length that satisfies the
            // modular identity yet dwarfs the validated region.
            let field_si = bfe!(2);
            let wrapped_len = (field_si - bfe!(1)) * elem_size.inverse();
            assert_eq!(field_si, wrapped_len * elem_size + bfe!(1));
            assert!(wrapped_len.value() > u32::MAX as u64);

            // Post-fix: a length exceeding `u32::MAX` is caught by the `lt`
            // instruction's u32 decomposition before the multiply.
            let result = run(field_si, wrapped_len);
            assert!(
                matches!(
                    result,
                    Err(InstructionError::OpStackError(
                        OpStackError::FailedU32Conversion(_)
                    ))
                ),
                "overlong static-element Vec length must be rejected, got {result:?}"
            );

            // A length that is a valid u32 but >= MAX_OFFSET is caught by the
            // explicit bound (`error_id 212`). `MAX_OFFSET` itself fails the
            // strict `<` check.
            let max_offset = bfe!(u64::from(DEFAULT_MAX_DYN_FIELD_SIZE));
            let result = run(bfe!(2), max_offset);
            assert!(
                matches!(result, Err(InstructionError::AssertionFailed(_))),
                "static-element Vec length >= MAX_OFFSET must be rejected, got {result:?}"
            );

            // Happy path: an honest single-`Digest` vector (field_si = 6, len = 1)
            // is still accepted and reports the correct struct size (7).
            let honest = run(bfe!(6), bfe!(1)).unwrap();
            assert_eq!(bfe!(7), *honest.op_stack.stack.last().unwrap());
        }
    }
}

#[cfg(test)]
mod benches {
    use super::*;
    use crate::neptune::neptune_like_types_for_tests::ProofCollectionLookalike;
    use crate::neptune::neptune_like_types_for_tests::TransactionKernelLookalike;
    use crate::test_prelude::*;

    #[macro_rules_attr::apply(test)]
    fn bench_proof_collection_lookalike() {
        ShadowedAccessor::new(VerifyNdSiIntegrity::<ProofCollectionLookalike>::default()).bench();
    }

    #[macro_rules_attr::apply(test)]
    fn bench_transaction_kernel_lookalike() {
        ShadowedAccessor::new(VerifyNdSiIntegrity::<TransactionKernelLookalike>::default()).bench();
    }
}
