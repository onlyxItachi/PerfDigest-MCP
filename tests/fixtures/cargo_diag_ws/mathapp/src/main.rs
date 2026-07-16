//! mathapp — tiny bin crate depending on mathlib, used ONLY to generate the
//! cargo_diag test fixtures. Compiles clean (zero warnings) in the warnings
//! fixture so one crate carries genuine-0.0 counts; the error fixture is
//! captured after a real type error is introduced here.

use mathlib::{add, mul};

fn main() {
    let a = add(2, 3);
    let m = mul(4, 5);
    println!("add(2,3)={a} mul(4,5)={m}");
}
