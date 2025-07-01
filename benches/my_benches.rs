// benches/my_benches.rs
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use glam::f32 as glam_f32;
use math_library::prelude::*; // Your crate // The competitor

fn normalize_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vec3 Normalize");

    let v = Vec3::new(1.2f32, 3.4, 5.6);
    let gv = glam_f32::Vec3::new(1.2f32, 3.4, 5.6);

    group.bench_with_input(BenchmarkId::new("My Crate", "Vec3"), &v, |b, vec| {
        b.iter(|| black_box(vec.normalize()))
    });

    group.bench_with_input(BenchmarkId::new("glam", "Vec3"), &gv, |b, vec| {
        b.iter(|| black_box(vec.normalize()))
    });

    group.finish();
}

fn mat4_mul_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mat4 Multiplication");

    let m1 = Mat4::<f32>::from_translation(Vec3::new(1.0, 2.0, 3.0));
    let m2 = Mat4::<f32>::from_scale(Vec3::new(4.0, 5.0, 6.0));

    let gm1 = glam_f32::Mat4::from_translation(glam_f32::Vec3::new(1.0, 2.0, 3.0));
    let gm2 = glam_f32::Mat4::from_scale(glam_f32::Vec3::new(4.0, 5.0, 6.0));

    group.bench_with_input(
        BenchmarkId::new("My Crate", "Mat4"),
        &(&m1, &m2),
        |b, (m1, m2)| b.iter(|| black_box(**m1 * **m2)),
    );

    group.bench_with_input(
        BenchmarkId::new("glam", "Mat4"),
        &(&gm1, &gm2),
        |b, (m1, m2)| b.iter(|| black_box(**m1 * **m2)),
    );

    group.finish();
}

fn normalize_edge_cases_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vec3 Normalize Edge Cases");

    let v_normal = Vec3::new(1.2f32, 3.4, 5.6);
    let v_zero = Vec3::<f32>::default();
    let v_inf = Vec3::new(f32::INFINITY, 1.0, 2.0);

    group.bench_function("Normal", |b| b.iter(|| black_box(v_normal.normalize())));
    group.bench_function("Zero", |b| b.iter(|| black_box(v_zero.normalize())));
    group.bench_function("Infinite", |b| b.iter(|| black_box(v_inf.normalize())));

    group.finish();
}

criterion_group!(
    benches,
    normalize_benchmark,
    mat4_mul_benchmark,
    normalize_edge_cases_benchmark
);
criterion_main!(benches);
