use neptune_privacy::api::export::Utxo;
use neptune_privacy::prelude::triton_vm::prelude::BFieldElement;
use neptune_privacy::prelude::twenty_first::tip5::digest::Digest;
use neptune_privacy::protocol::consensus::transaction::utxo::Coin;
use tasm_lib::twenty_first::prelude::Tip5;

#[test]
fn compute_target_utxo_digest() {
    let lock_script_hash = Digest::try_from_hex(
        "eced8e8e409c02617f1ba20572cc2400e6949cd0c2529c01a5566198218ff70138305645f241114c",
    )
    .expect("lock_script_hash hex");

    let native_currency = Coin {
        type_script_hash: Digest::try_from_hex(
            "f8e778e011688c3985f0a5b325e26a19d7d6dc2a50cb8356b569bd9acc98eabc8511bb707d0b7a64",
        )
        .expect("nc ts hash hex"),
        state: vec![
            BFieldElement::new(2147483648),
            BFieldElement::new(2364136404),
            BFieldElement::new(1046034848),
            BFieldElement::new(25),
        ],
    };

    let time_lock = Coin {
        type_script_hash: Digest::try_from_hex(
            "4b4d251947a07f9f2c016c1c271c04ce41013ff50031bd42854919be6e0e4849ebf931e856b542ad",
        )
        .expect("tl ts hash hex"),
        state: vec![BFieldElement::new(1_779_721_196_396)], // T+40 (14:59:56 UTC) — pivoted from T+25 because +1h window was closing too fast
    };

    let utxo = Utxo::new(lock_script_hash, vec![native_currency, time_lock]);
    let digest = Tip5::hash(&utxo);

    println!("TARGET_DIGEST hex = {}", digest.to_hex());
    println!("TARGET_DIGEST values = {:?}", digest.values());
}
