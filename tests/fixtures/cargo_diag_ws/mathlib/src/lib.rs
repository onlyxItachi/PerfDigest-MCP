//! mathlib — tiny lib crate used ONLY to generate the cargo_diag test
//! fixtures. Compiles clean but with deliberate warnings (dead code,
//! unused variable) so the compiler-message stream has real payload.

pub fn add(a: i64, b: i64) -> i64 {
    let unused_sum = a + b + 1; // deliberate `unused_variables` warning
    a + b
}

pub fn mul(a: i64, b: i64) -> i64 {
    a * b
}

// deliberate `dead_code` warning: private and never called
fn never_called(x: i64) -> i64 {
    x - 1
}
