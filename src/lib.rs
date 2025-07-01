//! # My Awesome Math Library
//!
//! A simple and fast 2D/3D/4D math library for graphics and games,
//! written as a learning exercise.
//!
//! ## Usage
//!
//! Add this to your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! math_library = "0.1.0"
//! ```
//!
//! And then you can use the types in your code:
//!
//! ```
//! use math_library::{Vec3, Mat4};
//!
//! let point = Vec3::new(1.0, 2.0, 3.0);
//! // ...
//! ```

// Declare the modules
pub mod matrix;
pub mod vector;

// Re-export the main types for easier access
pub use matrix::Mat4;
pub use vector::{Vec2, Vec3, Vec4};

pub mod prelude {
    pub use crate::matrix::Mat4;
    pub use crate::vector::{Vec2, Vec3, Vec4};
}
