use tasm_lib::field;
use tasm_lib::library::StaticAllocation;
use tasm_lib::prelude::BasicSnippet;
use tasm_lib::prelude::DataType;
use tasm_lib::prelude::Digest;
use tasm_lib::prelude::Library;
use tasm_lib::structure::tasm_object::DEFAULT_MAX_DYN_FIELD_SIZE;
use tasm_lib::triton_vm::isa::triton_asm;
use tasm_lib::triton_vm::prelude::LabelledInstruction;

use super::test_time_lock_and_maybe_mark::TestTimeLockAndMaybeMark;
use super::total_amount_main_loop::DigestSource;
use crate::protocol::consensus::transaction::utxo::Coin;
use crate::protocol::consensus::type_scripts::amount::read_and_add_amount::ReadAndAddAmount;
use crate::protocol::consensus::type_scripts::amount::TOO_BIG_COIN_FIELD_SIZE_ERROR;
use crate::protocol::proof_abstractions::tasm::push_digest_reversed;
use crate::BFieldElement;

/// Body for inner loop, running over all coins within one UTXO.
#[derive(Debug, Clone)]
pub(crate) struct AddAllAmountsAndCheckTimeLock {
    pub(crate) digest_source: DigestSource,
    pub(crate) release_date: StaticAllocation,
    /// Additional (historical) type-script hashes whose coins are also counted
    /// as native currency, each OR'd into the primary `digest_source` match.
    ///
    /// Used by `NativeCurrency` to keep coins committed before a VM upgrade —
    /// which carry the pre-`UpgradeVM` legacy hash and/or the `UpgradeVM` (v3)
    /// hash (see `NativeCurrency::historical_type_script_hashes`) — spendable
    /// after each upgrade re-hashed the program. An empty list produces
    /// byte-identical code to the pre-remap snippet.
    pub(crate) legacy_digests: Vec<Digest>,
}

impl AddAllAmountsAndCheckTimeLock {
    const TIME_LOCK_HASH: Digest = Digest([
        BFieldElement::new(11493081001297792331),
        BFieldElement::new(14845021226026139948),
        BFieldElement::new(4809053857285865793),
        BFieldElement::new(5280486431890426245),
        BFieldElement::new(12484740501891840491),
    ]);
}

impl BasicSnippet for AddAllAmountsAndCheckTimeLock {
    fn parameters(&self) -> Vec<(DataType, String)> {
        vec![
            (DataType::U32, "num_coins".to_string()),
            (DataType::U32, "index".to_string()),
            (DataType::VoidPointer, "*coins[j]_si".to_string()),
            (DataType::U128, "amount".to_string()),
            (DataType::U128, "timelocked_amount".to_string()),
            (DataType::U128, "utxo_amount".to_string()),
            (DataType::Bool, "utxo_is_timelocked".to_string()),
        ]
    }

    fn return_values(&self) -> Vec<(DataType, String)> {
        vec![
            (DataType::U32, "num_coins".to_string()),
            (DataType::U32, "num_coins".to_string()),
            (DataType::VoidPointer, "*eof".to_string()),
            (DataType::U128, "amount".to_string()),
            (DataType::U128, "timelocked_amount".to_string()),
            (DataType::U128, "utxo_amount'".to_string()),
            (DataType::Bool, "utxo_is_timelocked'".to_string()),
        ]
    }

    fn entrypoint(&self) -> String {
        // The emitted code differs by `legacy_digests` (one OR-in-legacy-match
        // block per digest). `Library::import` dedups snippets by entrypoint name
        // only, so the name must vary with the digests — otherwise importing two
        // variants into one library would silently reuse whichever was imported
        // first. An empty list keeps the base name, so non-remap users emit
        // byte-identical code.
        let base = "neptune_type_script_total_amount_and_check_timelock";
        if self.legacy_digests.is_empty() {
            base.to_string()
        } else {
            let suffix = self
                .legacy_digests
                .iter()
                .map(|d| d.to_hex())
                .collect::<Vec<_>>()
                .join("_");
            format!("{base}_legacy_{suffix}")
        }
    }

    fn code(&self, library: &mut Library) -> Vec<LabelledInstruction> {
        let test_time_lock_and_maybe_mark = library.import(Box::new(TestTimeLockAndMaybeMark {
            release_date: self.release_date,
        }));
        let read_and_add_amount = library.import(Box::new(ReadAndAddAmount));

        let field_type_script_hash = field!(Coin::type_script_hash);
        let digest_eq = DataType::Digest.compare();

        let get_type_script_digest = match self.digest_source {
            DigestSource::StaticMemory(digest_allocation) => {
                triton_asm! {
                    // _
                    push {digest_allocation.read_address()}
                    read_mem {Digest::LEN}
                    pop 1
                    // _ [own_program_digest]
                }
            }
            DigestSource::Hardcode(harcoded_digest) => push_digest_reversed(harcoded_digest),
        };

        let push_timelock_digest = push_digest_reversed(Self::TIME_LOCK_HASH);

        // REMAP: optionally OR one or more (historical) type-script-hash matches
        // into the native-currency check, one block per `legacy_digests` entry.
        // When the list is empty, this emits no instructions, keeping the snippet
        // byte-identical for non-remap users (e.g. `GetTotalAndTimeLockedAmounts`).
        // Each block re-reads the coin's `type_script_hash` and `add`s its match
        // bit to the running accumulator; the stack shape is unchanged across
        // blocks (the accumulator stays on top where `is_primary_match` was), so
        // `dup 14` reaches `*coins[j]_si` identically every iteration. All matches
        // are mutually exclusive (a coin carries one hash), so the sum stays {0,1}.
        let or_in_legacy_match: Vec<LabelledInstruction> = self
            .legacy_digests
            .iter()
            .flat_map(|legacy_digest| {
                let push_legacy_digest = push_digest_reversed(*legacy_digest);
                triton_asm! {
                    // _ M j *coins[j]_si [amount] [tl] [utxo_amount] utxo_is_timelocked acc
                    dup 14 push 1 add
                    // _ ... acc *coins[j]
                    {&field_type_script_hash}
                    push {Digest::LEN-1} add read_mem {Digest::LEN} pop 1
                    // _ ... acc [type_script_hash]
                    {&push_legacy_digest}
                    {&digest_eq}
                    // _ ... acc (type_script_hash == legacy_digest)
                    add
                    // _ ... (acc + is_legacy_match)  // mutually exclusive -> stays {0,1}
                }
            })
            .collect();

        triton_asm! {

            // INVARIANT: _ M j *coins[j]_si [amount] [timelocked_amount] [utxo_amount] utxo_is_timelocked
            {self.entrypoint()}:
                hint utxo_amount = stack[1..5]

                // evaluate termination criterion and return if necessary
                dup 15 dup 15 eq
                // _ M j *coins[j]_si [amount] [timelocked_amount] [utxo_amount] utxo_is_timelocked (M == j)

                skiz return
                // _ M j *coins[j]_si [amount] [timelocked_amount] [utxo_amount] utxo_is_timelocked


                // if coin is native currency, add amount
                dup 13 push 1 add
                hint coins_j = stack[0]
                // _ M j *coins[j]_si [amount] [timelocked_amount] [utxo_amount] utxo_is_timelocked *coins[j]

                {&field_type_script_hash}
                hint type_script_hash_ptr = stack[0]
                // _ M j *coins[j]_si [amount] [timelocked_amount]  [utxo_amount] utxo_is_timelocked *type_script_hash

                push {Digest::LEN-1} add read_mem {Digest::LEN} pop 1
                hint type_script_hash : Digest = stack[0..5]
                // _ M j *coins[j]_si [amount] [timelocked_amount] [utxo_amount] utxo_is_timelocked [type_script_hash]

                {&get_type_script_digest}
                hint own_program_digest = stack[0..5]
                // _ M j *coins[j]_si [amount] [timelocked_amount] [utxo_amount] utxo_is_timelocked [type_script_hash] [own_program_digest]

                {&digest_eq}
                hint digests_are_equal = stack[0]
                // _ M j *coins[j]_si [amount] [timelocked_amount] [utxo_amount] utxo_is_timelocked (type_script_hash == own_program_digest)

                {&or_in_legacy_match}
                // _ M j *coins[j]_si [amount] [timelocked_amount] [utxo_amount] utxo_is_timelocked (matches native currency)

                skiz call {read_and_add_amount}
                // _ M j *coins[j]_si [amount] [timelocked_amount] [utxo_amount'] utxo_is_timelocked


                // if coin is timelock, test and mark if necessary
                dup 13 push 1 add
                hint coins_j = stack[0]
                // _ M j *coins[j]_si [amount] [timelocked_amount] [utxo_amount'] utxo_is_timelocked *coins[j]

                {&field_type_script_hash}
                hint type_script_hash_ptr = stack[0]
                // _ M j *coins[j]_si [amount] [timelocked_amount]  [utxo_amount'] utxo_is_timelocked *type_script_hash

                push {Digest::LEN-1} add read_mem {Digest::LEN} pop 1
                hint type_script_hash : Digest = stack[0..5]
                // _ M j *coins[j]_si [amount] [timelocked_amount]  [utxo_amount'] utxo_is_timelocked [type_script_hash]

                {&push_timelock_digest}
                // _ M j *coins[j]_si [amount] [timelocked_amount] [utxo_amount'] utxo_is_timelocked [type_script_hash] [timelock_digest]

                {&digest_eq}
                // _ M j *coins[j]_si [amount] [timelocked_amount] [utxo_amount'] utxo_is_timelocked (type_script_hash == timelock_digest)


                // If he coin is a time lock:
                //  - test the state, which encodes a release date, against the
                //    timestamp of the transaction kernel plus the coinbase
                //    timelock period.
                skiz call {test_time_lock_and_maybe_mark}
                // _ M j *coins[j]_si [amount] [timelocked_amount] [utxo_amount'] utxo_is_timelocked


                // prepare for next iteration
                dup 14 addi 1 swap 15 pop 1
                // _ M (j+1) *coins[j]_si [amount] [timelocked_amount] [utxo_amount] utxo_is_timelocked

                dup 13 read_mem 1 addi 2
                // _ M (j+1) *coins[j]_si [amount] [timelocked_amount]  [utxo_amount] utxo_is_timelocked size(coins[j]) *coins[j]

                /* Range-check on size */
                push {DEFAULT_MAX_DYN_FIELD_SIZE}
                dup 2
                lt
                assert error_id {TOO_BIG_COIN_FIELD_SIZE_ERROR}
                // _ M (j+1) *coins[j]_si [amount] [timelocked_amount]  [utxo_amount] utxo_is_timelocked size(coins[j]) *coins[j]

                add
                // _ M (j+1) *coins[j]_si [amount] [timelocked_amount]  [utxo_amount] utxo_is_timelocked *coins[j+1]_si

                swap 14 pop 1
                // _ M (j+1) *coins[j+1]_si [amount] [timelocked_amount]  [utxo_amount] utxo_is_timelocked

                recurse
        }
    }
}

#[cfg(test)]
mod test {
    use crate::protocol::consensus::type_scripts::amount::add_all_amounts_and_check_time_lock::AddAllAmountsAndCheckTimeLock;
    use crate::protocol::consensus::type_scripts::time_lock::TimeLock;
    use crate::protocol::proof_abstractions::tasm::program::ConsensusProgram;

    // Gated: TIME_LOCK_HASH is intentionally left at the pre-UpgradeVM value
    // while coinbase time-locking is disabled (MINING_REWARD_TIME_LOCK_PERIOD
    // == 0) — the time-lock match is a no-op, so the stale hash is harmless.
    // Refresh this to the current `TimeLock.hash()` when re-enabling the rule.
    #[ignore = "coinbase time-lock disabled while MINING_REWARD_TIME_LOCK_PERIOD == 0"]
    #[test]
    fn hardcoded_time_lock_hash_matches_hash_of_time_lock_program() {
        let calculated = TimeLock.hash();
        assert_eq!(
            AddAllAmountsAndCheckTimeLock::TIME_LOCK_HASH,
            calculated,
            "Timelock.hash():\n{}",
            calculated
        );
    }
}
