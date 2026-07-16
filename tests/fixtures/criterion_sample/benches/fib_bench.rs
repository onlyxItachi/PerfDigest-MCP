use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion, SamplingMode};
use perfdigest_criterion_fixture::fib;

fn bench_plain(c: &mut Criterion) {
    // Linear sampling (criterion's default): estimates.json carries `slope`.
    c.bench_function("fib_plain", |b| b.iter(|| fib(std::hint::black_box(10))));
}

fn bench_grouped(c: &mut Criterion) {
    let mut group = c.benchmark_group("fib");
    // Flat sampling: criterion genuinely omits the `slope` estimator, giving
    // the honest-absence case with REAL data (nothing hand-trimmed).
    group.sampling_mode(SamplingMode::Flat);
    group.bench_function("fib_20", |b| b.iter(|| fib(std::hint::black_box(20))));
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_millis(400));
    targets = bench_plain, bench_grouped
}
criterion_main!(benches);
