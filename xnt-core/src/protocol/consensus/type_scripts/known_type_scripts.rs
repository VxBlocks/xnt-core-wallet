use tasm_lib::prelude::Digest;

use super::native_currency::NativeCurrency;
use super::native_currency::NativeCurrencyWitness;
use super::time_lock::TimeLock;
use super::time_lock::TimeLockWitness;
use super::time_lock_v2::TimeLockV2;
use super::time_lock_v2::TimeLockV2Witness;
use super::TypeScript;
use super::TypeScriptAndWitness;
use super::TypeScriptWitness;
use crate::protocol::consensus::transaction::primitive_witness::SaltedUtxos;
use crate::protocol::consensus::transaction::transaction_kernel::TransactionKernel;
use crate::protocol::consensus::transaction::utxo::Coin;
use crate::protocol::proof_abstractions::tasm::program::ConsensusProgram;

pub(crate) fn match_type_script_and_generate_witness(
    type_script_hash: Digest,
    transaction_kernel: TransactionKernel,
    salted_input_utxos: SaltedUtxos,
    salted_output_utxos: SaltedUtxos,
) -> Option<TypeScriptAndWitness> {
    let type_script_and_witness = if type_script_hash == NativeCurrency.hash() {
        NativeCurrencyWitness::new(transaction_kernel, salted_input_utxos, salted_output_utxos)
            .type_script_and_witness()
    } else if type_script_hash == TimeLock.hash() {
        TimeLockWitness::new(transaction_kernel, salted_input_utxos, salted_output_utxos)
            .type_script_and_witness()
    } else if type_script_hash == TimeLockV2.hash() {
        TimeLockV2Witness::new(transaction_kernel, salted_input_utxos, salted_output_utxos)
            .type_script_and_witness()
    } else {
        return None;
    };
    Some(type_script_and_witness)
}

/// Post-fork variant of [`match_type_script_and_generate_witness`].
///
/// Identical to the standard dispatch except that coins still tagged with the
/// legacy `TimeLock` hash are matched to a `TimeLockV2` witness. This mirrors
/// the on-chain remap performed by `CollectTypeScriptsV2`: post-fork, the
/// SingleProof claim for the (remapped) `TimeLockV2.hash()` must be backed by
/// a `TimeLockV2` proof even when the coin in question carries the legacy hash.
///
/// Callers should choose between this and the V1 dispatch based on the
/// consensus rule set in effect at the block height being constructed/verified.
pub(crate) fn match_type_script_and_generate_witness_v2(
    type_script_hash: Digest,
    transaction_kernel: TransactionKernel,
    salted_input_utxos: SaltedUtxos,
    salted_output_utxos: SaltedUtxos,
) -> Option<TypeScriptAndWitness> {
    let type_script_and_witness = if type_script_hash == NativeCurrency.hash() {
        NativeCurrencyWitness::new(transaction_kernel, salted_input_utxos, salted_output_utxos)
            .type_script_and_witness()
    } else if type_script_hash == TimeLock.hash() || type_script_hash == TimeLockV2.hash() {
        TimeLockV2Witness::new(transaction_kernel, salted_input_utxos, salted_output_utxos)
            .type_script_and_witness()
    } else {
        return None;
    };
    Some(type_script_and_witness)
}

pub(crate) fn is_known_type_script_with_valid_state(coin: &Coin) -> bool {
    NativeCurrency.matches_coin(coin)
        || TimeLock.matches_coin(coin)
        || TimeLockV2.matches_coin(coin)
}

pub(crate) fn typescript_name(type_script_hash: Digest) -> &'static str {
    if type_script_hash == NativeCurrency.hash() {
        "native currency"
    } else if type_script_hash == TimeLock.hash() {
        "time lock"
    } else if type_script_hash == TimeLockV2.hash() {
        "time lock v2"
    } else {
        "unknown"
    }
}
