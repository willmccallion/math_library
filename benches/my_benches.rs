use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use glam::f32 as glam_f32;
use mathtools::prelude::*;

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
    let mut group = c.benchmark_group("Mat4 * Mat4 (Matrix Multiplication)");

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

fn dot_product_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vec3 Dot Product");

    let v1 = Vec3::new(1.2f32, 3.4, 5.6);
    let v2 = Vec3::new(7.8f32, 9.0, 1.2);
    let gv1 = glam_f32::Vec3::new(1.2f32, 3.4, 5.6);
    let gv2 = glam_f32::Vec3::new(7.8f32, 9.0, 1.2);

    group.bench_function("My Crate", |b| b.iter(|| black_box(v1.dot(v2))));
    group.bench_function("glam", |b| b.iter(|| black_box(gv1.dot(gv2))));

    group.finish();
}

fn cross_product_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vec3 Cross Product");

    let v1 = Vec3::new(1.2f32, 3.4, 5.6);
    let v2 = Vec3::new(7.8f32, 9.0, 1.2);
    let gv1 = glam_f32::Vec3::new(1.2f32, 3.4, 5.6);
    let gv2 = glam_f32::Vec3::new(7.8f32, 9.0, 1.2);

    group.bench_function("My Crate", |b| b.iter(|| black_box(v1.cross(v2))));
    group.bench_function("glam", |b| b.iter(|| black_box(gv1.cross(gv2))));

    group.finish();
}

fn matrix_vector_mul_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mat4 * Vec4 (Transform Point)");

    let m = Mat4::<f32>::from_axis_angle(Vec3::new(0.2, 0.4, 0.6), 1.5);
    let v = Vec4::new(10.0, 20.0, 30.0, 1.0);

    let gm = glam_f32::Mat4::from_axis_angle(glam_f32::Vec3::new(0.2, 0.4, 0.6), 1.5);
    let gv = glam_f32::Vec4::new(10.0, 20.0, 30.0, 1.0);

    group.bench_function("My Crate", |b| b.iter(|| black_box(m * v)));
    group.bench_function("glam", |b| b.iter(|| black_box(gm * gv)));

    group.finish();
}

fn matrix_inversion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mat4 Inverse");

    let m = Mat4::<f32>::from_translation(Vec3::new(10.0, -20.0, 30.0))
        * Mat4::from_axis_angle(Vec3::new(0.2, 0.4, 0.6), 1.5);

    let gm = glam_f32::Mat4::from_translation(glam_f32::Vec3::new(10.0, -20.0, 30.0))
        * glam_f32::Mat4::from_axis_angle(glam_f32::Vec3::new(0.2, 0.4, 0.6), 1.5);

    group.bench_function("My Crate", |b| b.iter(|| black_box(m.inverse())));
    group.bench_function("glam", |b| b.iter(|| black_box(gm.inverse())));

    group.finish();
}

criterion_group!(
    benches,
    normalize_benchmark,
    mat4_mul_benchmark,
    normalize_edge_cases_benchmark,
    dot_product_benchmark,
    cross_product_benchmark,
    matrix_vector_mul_benchmark,
    matrix_inversion_benchmark
);

criterion_main!(benches);
