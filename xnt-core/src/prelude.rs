//! Re-exports the most commonly-needed APIs of xnt-core.
//!
//! This module is intended to be wildcard-imported, _i.e._, `use neptune_privacy::prelude::twenty_first;`.
//! You might also want to consider wildcard-importing these prelude,
//! `use neptune_privacy::prelude::tasm_lib::prelude::*;`.
//! `use neptune_privacy::prelude::triton_vm::prelude::*;`.

pub use tasm_lib;
pub use tasm_lib::prelude::triton_vm;
pub use tasm_lib::prelude::twenty_first;
