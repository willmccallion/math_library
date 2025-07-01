# mathtools

<!--
-->

[![Crates.io](https://img.shields.io/crates/v/mathtools.svg)](https://crates.io/crates/mathtools)
[![Docs.rs](https://docs.rs/mathtools/badge.svg)](https://docs.rs/mathtools)
[![Licence](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#licence)

<!--
[![CI](https://github.com/willmccallion/mathtools/actions/workflows/ci.yml/badge.svg)](https://github.com/willmccallion/mathtools/actions/workflows/ci.yml)
-->

A lightweight, portable, and performant linear algebra library for graphics and game development, written in pure, stable Rust.

This library provides `Vec2`, `Vec3`, `Vec4`, and `Mat4` types with a rich and ergonomic API suitable for real-time applications. It is designed with a "batteries-included" philosophy, offering the features you need for 2D/3D math without unnecessary complexity or unsafe code.

## Key Features

* **⚡ Performant:** Best-in-class scalar performance. For fundamental vector operations like dot and cross products, `mathtools` is on par with or faster than top-tier SIMD libraries. See the [Performance](#performance) section for details.
* **📦 Portable & `no_std`:** Fully compatible with `no_std` environments for use in embedded systems, WebAssembly, and OS development. The library is `no_std` by default.
* **🎛️ Optional `serde` Support:** Easily serialize and deserialize all vector and matrix types by enabling the `serde` feature.
* **🧑‍💻 Ergonomic API:** Uses standard Rust operators (`+`, `-`, `*`) for intuitive vector and matrix arithmetic. The API is designed to be clear, consistent, and easy to use.
* **✅ Safe & Robust:** Written in 100% safe, stable Rust. The comprehensive test suite covers correctness, mathematical properties, and floating-point edge cases to ensure reliability.

## Usage & Features

Add `mathtools` to your `Cargo.toml`. Here are some common configurations:

### For `std` environments (Recommended for most applications)

```toml
[dependencies]
mathtools = { version = "0.1.0", features = ["std"] }
```

### For `no_std` environments

```toml
[dependencies]
mathtools = { version = "0.1.0", default-features = false }
```

### With `serde` Support

#### For std environments

```toml
[dependencies]
mathtools = { version = "0.1.0", features = ["std", "serde"] }
```

#### For no\_std environments

```toml
[dependencies]
mathtools = { version = "0.1.0", default-features = false, features = ["serde"] }
```

Then, import the types you need, ideally through the provided prelude:

```rust
use mathtools::prelude::*;
```

### Quick Start Vector Operations

```rust
use mathtools::prelude::*;

// Vector creation and arithmetic
let v1 = Vec2::new(1.0f32, 2.0);
let v2 = Vec2::splat(10.0);
let v3 = (v1 + v2) * 2.0; // (11.0, 12.0) * 2.0 = (22.0, 24.0)

assert_eq!(v3, Vec2::new(22.0, 24.0));

// Dot and cross products
let i = Vec3::new(1.0, 0.0, 0.0);
let j = Vec3::new(0.0, 1.0, 0.0);

assert_eq!(i.dot(j), 0.0);
assert_eq!(i.cross(j), Vec3::new(0.0, 0.0, 1.0)); // The k vector

// Normalization
let v = Vec3::new(3.0f32, 4.0, 0.0);
assert_eq!(v.length(), 5.0);

let norm_v = v.normalize();
assert!((norm_v.length() - 1.0).abs() < 1e-6);
```

### 3D Transformations

`Mat4` can be used to build complex 3D transformations. Transformations are applied from right to left (`translation * rotation * scale * point`)

```rust
use mathtools::prelude::*;
use core::f32::consts::FRAC_PI_2; // 90 degrees

// 1. A point to be transformed.
let point = Vec4::new(1.0, 0.0, 0.0, 1.0); // A point at (1,0,0)

// 2. Create a rotation matrix for a 90-degree turn around the Y axis.
let rotation = Mat4::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), FRAC_PI_2);

// 3. Create a translation matrix to move 10 units along the X axis.
let translation = Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0));

// 4. Combine the transformations: first rotate, then translate.
let transform = translation * rotation;

// 5. Apply the combined transformation to the point.
let transformed_point = transform * point;

// The point (1,0,0) is first rotated to (0,0,-1), then translated to (10,0,-1).
let expected_point = Vec4::new(10.0, 0.0, -1.0, 1.0);

// Use an approximate comparison for floating-point results
assert!((transformed_point.x - expected_point.x).abs() < 1e-6);
assert!((transformed_point.y - expected_point.y).abs() < 1e-6);
assert!((transformed_point.z - expected_point.z).abs() < 1e-6);
```

### Performance

This library provides excellent scalar performance. Benchmarks against `glam` (a popular SIMD-accelerated library) show:

* **Vector Operations:** `mathtools` is on par with or faster than `glam` for fundamental operations like dot product, cross product, and normalization.
* **Matrix Operations:** For complex, parallelizable operations like matrix multiplication and inversion, `glam`'s SIMD implementation provides an expected speedup. `mathtools` still offers highly competitive performance for a portable, scalar library.

This makes `mathtools` an ideal choice where portability and a simple, robust design are priorities.

## Development

This library is equipped with a full test and benchmark suite.

**Run all tests for all feature combinations:**

```bash
# Test default (no_std)
cargo test --no-default-features
# Test no_std + serde
cargo test --no-default-features --features serde
# Test std
cargo test --features std
# Test std + serde
cargo test --all-features
```

**Run benchmarks (requires `std`):**

```bash
cargo bench
```

## Licence

This project is dual-licensed under your choice of either the [MIT License](https://github.com/willmccallion/mathtools/blob/main/LICENCE-MIT) or the [Apache License, Version 2.0](https://github.com/willmccallion/mathtools/blob/main/LICENCE-APACHE).

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

