#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate libm;

// Declare the modules
pub mod matrix;
pub mod vector;
pub mod prelude {
    pub use crate::matrix::Mat4;
    pub use crate::vector::{Vec2, Vec3, Vec4};
}

// Re-export the main types for easier access
pub use matrix::Mat4;
pub use vector::{Vec2, Vec3, Vec4};
