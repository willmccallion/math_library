#![doc = include_str!("../README.md")]

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
