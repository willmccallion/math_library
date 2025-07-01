# math_library

<!-- Badges: Replace placeholders when you publish and set up CI -->
[![Crates.io](https://img.shields.io/crates/v/math_library.svg)](https://crates.io/crates/math_library)
[![Docs.rs](https://docs.rs/math_library/badge.svg)](https://docs.rs/math_library)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)
<!--
[![CI](https://github.com/willmccallion/math_library/actions/workflows/ci.yml/badge.svg)](https://github.com/willmccallion/math_library/actions/workflows/ci.yml)
-->

A simple and fast 2D/3D/4D math library for graphics and games, written in pure, stable Rust.

This library provides `Vec2`, `Vec3`, `Vec4`, and `Mat4` types with a rich and ergonomic API suitable for real-time applications. It is designed to be lightweight, with no dependencies outside of `num-traits`.

## Features

-   **Vector Types**: `Vec2`, `Vec3`, and `Vec4` with a full suite of mathematical functions.
-   **Matrix Type**: A column-major `Mat4` for 3D transformations.
-   **Ergonomic API**: Uses standard Rust operators (`+`, `-`, `*`) for intuitive vector and matrix arithmetic.
-   **Rich Functionality**: Includes constructors for translation, rotation, scale, `look_at` view matrices, and perspective projections.
-   **Robust and Safe**: Written in 100% safe, stable Rust. No `unsafe` code.
-   **Well-Tested**: Includes a comprehensive suite of unit and documentation tests to ensure correctness.
-   **Fully Documented**: Every public function is documented with examples. Run `cargo doc --open` to view them.

## Usage

Add `math_library` to your `Cargo.toml` file:

```toml
[dependencies]
math_library = "0.1.0" # Replace with the latest version from crates.io
```

Then, you can use the types in your project:

```rust
use math_library::{Vec3, Mat4};

// ...
```

## Examples

### Vector Operations

The vector types support standard arithmetic operations, as well as common functions like `dot` product, `cross` product, and `normalize`.

```rust
use math_library::{Vec2, Vec3};

// Vector creation and arithmetic
let v1 = Vec2::new(1.0, 2.0);
let v2 = Vec2::splat(10.0);
let v3 = (v1 + v2) * 2.0; // (11.0, 12.0) * 2.0 = (22.0, 24.0)

assert_eq!(v3, Vec2::new(22.0, 24.0));

// Dot and cross products
let i = Vec3::new(1.0, 0.0, 0.0);
let j = Vec3::new(0.0, 1.0, 0.0);

assert_eq!(i.dot(j), 0.0);
assert_eq!(i.cross(j), Vec3::new(0.0, 0.0, 1.0)); // k

// Normalization
let v = Vec3::<f32>::new(3.0, 4.0, 0.0);
assert_eq!(v.length(), 5.0);

let norm_v = v.normalize();
assert!((norm_v.length() - 1.0).abs() < 1e-6);
```

### Matrix Transformations

`Mat4` can be used to build complex 3D transformations by combining simpler ones. The order of multiplication matters! In this library, transformations are applied from right to left (`translate * rotate * scale * point`).

```rust
use math_library::{Mat4, Vec3, Vec4};
use std::f32::consts::FRAC_PI_2; // 90 degrees

// 1. A point to be transformed.
let point = Vec4::new(1.0, 0.0, 0.0, 1.0); // A point at (1,0,0)

// 2. Create a rotation matrix for a 90-degree turn around the Y axis.
let rotation = Mat4::from_axis_angle(
    Vec3::new(0.0, 1.0, 0.0),
    FRAC_PI_2
);

// 3. Create a translation matrix to move 10 units along the X axis.
let translation = Mat4::from_translation(
    Vec3::new(10.0, 0.0, 0.0)
);

// 4. Combine the transformations: first rotate, then translate.
let transform = translation * rotation;

// 5. Apply the combined transformation to the point.
let transformed_point = transform * point;

// The point (1,0,0) is first rotated to (0,0,-1),
// then translated to (10,0,-1).
let expected_point = Vec4::new(10.0, 0.0, -1.0, 1.0);

assert!((transformed_point.x - expected_point.x).abs() < 1e-6);
assert!((transformed_point.y - expected_point.y).abs() < 1e-6);
assert!((transformed_point.z - expected_point.z).abs() < 1e-6);
```

## Running Tests

This library is equipped with a full test suite. To run all tests:

```bash
# Run unit tests
cargo test

# Run documentation tests as well
cargo test --doc
```

## License

This project is dual-licensed under your choice of either the [MIT License](LICENCE-MIT) or the [Apache License, Version 2.0](LICENCE-APACHE).

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
