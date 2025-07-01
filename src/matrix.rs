use crate::vector::{Vec3, Vec4};
use num_traits::{Float, One, Zero};
use std::ops::{Add, Mul, MulAssign};

/// A 4x4, column-major matrix.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4<T> {
    pub col1: Vec4<T>,
    pub col2: Vec4<T>,
    pub col3: Vec4<T>,
    pub col4: Vec4<T>,
}

impl<T: Copy + One + Zero> Mat4<T> {
    /// Creates a new matrix from four column vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// # use math_library::{Mat4, Vec4};
    /// let m = Mat4::new(
    ///     Vec4::new(1i32, 2, 3, 4),
    ///     Vec4::new(5, 6, 7, 8),
    ///     Vec4::new(9, 10, 11, 12),
    ///     Vec4::new(13, 14, 15, 16),
    /// );
    /// assert_eq!(m.col1.y, 2);
    /// ```
    pub fn new(col1: Vec4<T>, col2: Vec4<T>, col3: Vec4<T>, col4: Vec4<T>) -> Self {
        Self {
            col1,
            col2,
            col3,
            col4,
        }
    }

    /// Creates an identity matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use math_library::{Mat4, Vec4};
    /// let id = Mat4::<f32>::identity();
    /// let v = Vec4::new(1.0, 2.0, 3.0, 1.0);
    /// assert_eq!(id * v, v);
    /// ```
    pub fn identity() -> Self {
        Self {
            col1: Vec4::new(T::one(), T::zero(), T::zero(), T::zero()),
            col2: Vec4::new(T::zero(), T::one(), T::zero(), T::zero()),
            col3: Vec4::new(T::zero(), T::zero(), T::one(), T::zero()),
            col4: Vec4::new(T::zero(), T::zero(), T::zero(), T::one()),
        }
    }
}

impl<T: Copy + One + Zero> Default for Mat4<T> {
    /// Returns an identity matrix.
    fn default() -> Self {
        Self::identity()
    }
}

impl<T: Float> Mat4<T> {
    /// Creates a translation matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use math_library::{Mat4, Vec3, Vec4};
    /// let t = Mat4::from_translation(Vec3::new(10.0f32, 20.0, 30.0));
    /// let p = Vec4::new(1.0, 2.0, 3.0, 1.0);
    /// let p_translated = t * p;
    /// assert_eq!(p_translated, Vec4::new(11.0, 22.0, 33.0, 1.0));
    /// ```
    pub fn from_translation(translation: Vec3<T>) -> Self {
        Self {
            col4: translation.to_vec4_point(),
            ..Self::identity()
        }
    }

    /// Creates a scaling matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use math_library::{Mat4, Vec3, Vec4};
    /// let s = Mat4::from_scale(Vec3::new(2.0f32, 3.0, 4.0));
    /// let v = Vec4::new(1.0, 2.0, 3.0, 1.0);
    /// let v_scaled = s * v;
    /// assert_eq!(v_scaled, Vec4::new(2.0, 6.0, 12.0, 1.0));
    /// ```
    pub fn from_scale(scale: Vec3<T>) -> Self {
        Self {
            col1: Vec4::new(scale.x, T::zero(), T::zero(), T::zero()),
            col2: Vec4::new(T::zero(), scale.y, T::zero(), T::zero()),
            col3: Vec4::new(T::zero(), T::zero(), scale.z, T::zero()),
            col4: Vec4::new(T::zero(), T::zero(), T::zero(), T::one()),
        }
    }

    /// Creates a rotation matrix around an arbitrary axis.
    ///
    /// # Examples
    ///
    /// ```
    /// # use math_library::{Mat4, Vec3, Vec4};
    /// # use std::f32::consts::FRAC_PI_2;
    /// let r = Mat4::from_axis_angle(Vec3::new(0.0f32, 1.0, 0.0), FRAC_PI_2);
    /// let v = Vec4::new(1.0, 0.0, 0.0, 1.0);
    /// let v_rotated = r * v;
    /// let expected = Vec4::new(0.0, 0.0, -1.0, 1.0);
    /// assert!((v_rotated.x - expected.x).abs() < 1e-6);
    /// assert!((v_rotated.z - expected.z).abs() < 1e-6);
    /// ```
    pub fn from_axis_angle(axis: Vec3<T>, angle: T) -> Self {
        let (s, c) = angle.sin_cos();
        let axis = axis.normalize();
        let t = T::one() - c;
        let x = axis.x;
        let y = axis.y;
        let z = axis.z;
        Self {
            col1: Vec4::new(
                t * x * x + c,
                t * x * y + s * z,
                t * x * z - s * y,
                T::zero(),
            ),
            col2: Vec4::new(
                t * x * y - s * z,
                t * y * y + c,
                t * y * z + s * x,
                T::zero(),
            ),
            col3: Vec4::new(
                t * x * z + s * y,
                t * y * z - s * x,
                t * z * z + c,
                T::zero(),
            ),
            col4: Vec4::new(T::zero(), T::zero(), T::zero(), T::one()),
        }
    }

    /// Creates a view matrix that looks from `eye` towards `target` with a given `up` direction.
    ///
    /// # Examples
    ///
    /// ```
    /// # use math_library::{Mat4, Vec3, Vec4};
    /// let eye = Vec3::new(0.0f32, 0.0, 5.0);
    /// let target = Vec3::new(0.0, 0.0, 0.0);
    /// let up = Vec3::new(0.0, 1.0, 0.0);
    ///
    /// let view_matrix = Mat4::look_at(eye, target, up);
    ///
    /// let point_at_origin = Vec4::new(0.0, 0.0, 0.0, 1.0);
    /// let transformed_point = view_matrix * point_at_origin;
    /// assert_eq!(transformed_point, Vec4::new(0.0, 0.0, -5.0, 1.0));
    /// ```
    pub fn look_at(eye: Vec3<T>, target: Vec3<T>, up: Vec3<T>) -> Self {
        let f = (target - eye).normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(f);

        Self {
            col1: Vec4::new(s.x, u.x, -f.x, T::zero()),
            col2: Vec4::new(s.y, u.y, -f.y, T::zero()),
            col3: Vec4::new(s.z, u.z, -f.z, T::zero()),
            col4: Vec4::new(-s.dot(eye), -u.dot(eye), f.dot(eye), T::one()),
        }
    }

    /// Creates a perspective projection matrix.
    pub fn perspective(fovy: T, aspect_ratio: T, near: T, far: T) -> Self {
        let f = T::one() / (fovy / (T::one() + T::one())).tan();
        Self {
            col1: Vec4::new(f / aspect_ratio, T::zero(), T::zero(), T::zero()),
            col2: Vec4::new(T::zero(), f, T::zero(), T::zero()),
            col3: Vec4::new(T::zero(), T::zero(), (far + near) / (near - far), -T::one()),
            col4: Vec4::new(
                T::zero(),
                T::zero(),
                (T::one() + T::one()) * far * near / (near - far),
                T::zero(),
            ),
        }
    }
}

impl<T: Float> Mat4<T> {
    /// Returns the transpose of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use math_library::{Mat4, Vec4};
    /// let m = Mat4::new(
    ///     Vec4::new(1.0f32, 2.0, 3.0, 4.0),
    ///     Vec4::new(5.0, 6.0, 7.0, 8.0),
    ///     Vec4::new(9.0, 10.0, 11.0, 12.0),
    ///     Vec4::new(13.0, 14.0, 15.0, 16.0),
    /// );
    /// let t = m.transpose();
    /// assert_eq!(t.col2.x, 2.0);
    /// assert_eq!(t.col1.y, 5.0);
    /// ```
    #[inline]
    pub fn transpose(&self) -> Self {
        Self {
            col1: Vec4::new(self.col1.x, self.col2.x, self.col3.x, self.col4.x),
            col2: Vec4::new(self.col1.y, self.col2.y, self.col3.y, self.col4.y),
            col3: Vec4::new(self.col1.z, self.col2.z, self.col3.z, self.col4.z),
            col4: Vec4::new(self.col1.w, self.col2.w, self.col3.w, self.col4.w),
        }
    }

    /// Returns the inverse of the matrix. Returns `None` if the matrix is not invertible.
    ///
    /// # Examples
    ///
    /// ```
    /// # use math_library::{Mat4, Vec3};
    /// let m = Mat4::from_translation(Vec3::new(10.0f32, -5.0, 2.0));
    /// let m_inv = m.inverse().unwrap();
    /// let identity = m * m_inv;
    ///
    /// assert_eq!(identity, Mat4::identity());
    /// ```
    pub fn inverse(&self) -> Option<Self> {
        let m = self.to_2d_array();
        let mut inv = [[T::zero(); 4]; 4];

        inv[0][0] =
            m[1][1] * m[2][2] * m[3][3] - m[1][1] * m[2][3] * m[3][2] - m[2][1] * m[1][2] * m[3][3]
                + m[2][1] * m[1][3] * m[3][2]
                + m[3][1] * m[1][2] * m[2][3]
                - m[3][1] * m[1][3] * m[2][2];
        inv[1][0] = -m[1][0] * m[2][2] * m[3][3]
            + m[1][0] * m[2][3] * m[3][2]
            + m[2][0] * m[1][2] * m[3][3]
            - m[2][0] * m[1][3] * m[3][2]
            - m[3][0] * m[1][2] * m[2][3]
            + m[3][0] * m[1][3] * m[2][2];
        inv[2][0] =
            m[1][0] * m[2][1] * m[3][3] - m[1][0] * m[2][3] * m[3][1] - m[2][0] * m[1][1] * m[3][3]
                + m[2][0] * m[1][3] * m[3][1]
                + m[3][0] * m[1][1] * m[2][3]
                - m[3][0] * m[1][3] * m[2][1];
        inv[3][0] = -m[1][0] * m[2][1] * m[3][2]
            + m[1][0] * m[2][2] * m[3][1]
            + m[2][0] * m[1][1] * m[3][2]
            - m[2][0] * m[1][2] * m[3][1]
            - m[3][0] * m[1][1] * m[2][2]
            + m[3][0] * m[1][2] * m[2][1];
        inv[0][1] = -m[0][1] * m[2][2] * m[3][3]
            + m[0][1] * m[2][3] * m[3][2]
            + m[2][1] * m[0][2] * m[3][3]
            - m[2][1] * m[0][3] * m[3][2]
            - m[3][1] * m[0][2] * m[2][3]
            + m[3][1] * m[0][3] * m[2][2];
        inv[1][1] =
            m[0][0] * m[2][2] * m[3][3] - m[0][0] * m[2][3] * m[3][2] - m[2][0] * m[0][2] * m[3][3]
                + m[2][0] * m[0][3] * m[3][2]
                + m[3][0] * m[0][2] * m[2][3]
                - m[3][0] * m[0][3] * m[2][2];
        inv[2][1] = -m[0][0] * m[2][1] * m[3][3]
            + m[0][0] * m[2][3] * m[3][1]
            + m[2][0] * m[0][1] * m[3][3]
            - m[2][0] * m[0][3] * m[3][1]
            - m[3][0] * m[0][1] * m[2][3]
            + m[3][0] * m[0][3] * m[2][1];
        inv[3][1] =
            m[0][0] * m[2][1] * m[3][2] - m[0][0] * m[2][2] * m[3][1] - m[2][0] * m[0][1] * m[3][2]
                + m[2][0] * m[0][2] * m[3][1]
                + m[3][0] * m[0][1] * m[2][2]
                - m[3][0] * m[0][2] * m[2][1];
        inv[0][2] =
            m[0][1] * m[1][2] * m[3][3] - m[0][1] * m[1][3] * m[3][2] - m[1][1] * m[0][2] * m[3][3]
                + m[1][1] * m[0][3] * m[3][2]
                + m[3][1] * m[0][2] * m[1][3]
                - m[3][1] * m[0][3] * m[1][2];
        inv[1][2] = -m[0][0] * m[1][2] * m[3][3]
            + m[0][0] * m[1][3] * m[3][2]
            + m[1][0] * m[0][2] * m[3][3]
            - m[1][0] * m[0][3] * m[3][2]
            - m[3][0] * m[0][2] * m[1][3]
            + m[3][0] * m[0][3] * m[1][2];
        inv[2][2] =
            m[0][0] * m[1][1] * m[3][3] - m[0][0] * m[1][3] * m[3][1] - m[1][0] * m[0][1] * m[3][3]
                + m[1][0] * m[0][3] * m[3][1]
                + m[3][0] * m[0][1] * m[1][3]
                - m[3][0] * m[0][3] * m[1][1];
        inv[3][2] = -m[0][0] * m[1][1] * m[3][2]
            + m[0][0] * m[1][2] * m[3][1]
            + m[1][0] * m[0][1] * m[3][2]
            - m[1][0] * m[0][2] * m[3][1]
            - m[3][0] * m[0][1] * m[1][2]
            + m[3][0] * m[0][2] * m[1][1];
        inv[0][3] = -m[0][1] * m[1][2] * m[2][3]
            + m[0][1] * m[1][3] * m[2][2]
            + m[1][1] * m[0][2] * m[2][3]
            - m[1][1] * m[0][3] * m[2][2]
            - m[2][1] * m[0][2] * m[1][3]
            + m[2][1] * m[0][3] * m[1][2];
        inv[1][3] =
            m[0][0] * m[1][2] * m[2][3] - m[0][0] * m[1][3] * m[2][2] - m[1][0] * m[0][2] * m[2][3]
                + m[1][0] * m[0][3] * m[2][2]
                + m[2][0] * m[0][2] * m[1][3]
                - m[2][0] * m[0][3] * m[1][2];
        inv[2][3] = -m[0][0] * m[1][1] * m[2][3]
            + m[0][0] * m[1][3] * m[2][1]
            + m[1][0] * m[0][1] * m[2][3]
            - m[1][0] * m[0][3] * m[2][1]
            - m[2][0] * m[0][1] * m[1][3]
            + m[2][0] * m[0][3] * m[1][1];
        inv[3][3] =
            m[0][0] * m[1][1] * m[2][2] - m[0][0] * m[1][2] * m[2][1] - m[1][0] * m[0][1] * m[2][2]
                + m[1][0] * m[0][2] * m[2][1]
                + m[2][0] * m[0][1] * m[1][2]
                - m[2][0] * m[0][2] * m[1][1];

        let det =
            m[0][0] * inv[0][0] + m[0][1] * inv[1][0] + m[0][2] * inv[2][0] + m[0][3] * inv[3][0];

        if det.abs() < T::epsilon() {
            return None;
        }

        let inv_det = T::one() / det;
        Some(Mat4::new(
            Vec4::new(
                inv[0][0] * inv_det,
                inv[1][0] * inv_det,
                inv[2][0] * inv_det,
                inv[3][0] * inv_det,
            ),
            Vec4::new(
                inv[0][1] * inv_det,
                inv[1][1] * inv_det,
                inv[2][1] * inv_det,
                inv[3][1] * inv_det,
            ),
            Vec4::new(
                inv[0][2] * inv_det,
                inv[1][2] * inv_det,
                inv[2][2] * inv_det,
                inv[3][2] * inv_det,
            ),
            Vec4::new(
                inv[0][3] * inv_det,
                inv[1][3] * inv_det,
                inv[2][3] * inv_det,
                inv[3][3] * inv_det,
            ),
        ))
    }

    fn to_2d_array(&self) -> [[T; 4]; 4] {
        [
            [self.col1.x, self.col2.x, self.col3.x, self.col4.x],
            [self.col1.y, self.col2.y, self.col3.y, self.col4.y],
            [self.col1.z, self.col2.z, self.col3.z, self.col4.z],
            [self.col1.w, self.col2.w, self.col3.w, self.col4.w],
        ]
    }
}

impl<T: Copy + Mul<Output = T> + Add<Output = T>> Mul<Vec4<T>> for Mat4<T> {
    type Output = Vec4<T>;
    #[inline]
    fn mul(self, rhs: Vec4<T>) -> Self::Output {
        let x =
            self.col1.x * rhs.x + self.col2.x * rhs.y + self.col3.x * rhs.z + self.col4.x * rhs.w;
        let y =
            self.col1.y * rhs.x + self.col2.y * rhs.y + self.col3.y * rhs.z + self.col4.y * rhs.w;
        let z =
            self.col1.z * rhs.x + self.col2.z * rhs.y + self.col3.z * rhs.z + self.col4.z * rhs.w;
        let w =
            self.col1.w * rhs.x + self.col2.w * rhs.y + self.col3.w * rhs.z + self.col4.w * rhs.w;
        Vec4::new(x, y, z, w)
    }
}

impl<T: Copy + Mul<Output = T> + Add<Output = T>> Mul<Mat4<T>> for Mat4<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            col1: self * rhs.col1,
            col2: self * rhs.col2,
            col3: self * rhs.col3,
            col4: self * rhs.col4,
        }
    }
}

impl<T: Copy + Mul<Output = T> + Add<Output = T>> MulAssign<Mat4<T>> for Mat4<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: Copy + Mul<Output = T>> Mul<T> for Mat4<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        Self {
            col1: self.col1 * rhs,
            col2: self.col2 * rhs,
            col3: self.col3 * rhs,
            col4: self.col4 * rhs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::Vec3;
    const EPSILON: f32 = 1e-5;

    fn assert_vec4_approx_eq(a: Vec4<f32>, b: Vec4<f32>) {
        assert!((a.x - b.x).abs() < EPSILON, "x: {} vs {}", a.x, b.x);
        assert!((a.y - b.y).abs() < EPSILON, "y: {} vs {}", a.y, b.y);
        assert!((a.z - b.z).abs() < EPSILON, "z: {} vs {}", a.z, b.z);
        assert!((a.w - b.w).abs() < EPSILON, "w: {} vs {}", a.w, b.w);
    }

    fn assert_mat4_approx_eq(a: Mat4<f32>, b: Mat4<f32>) {
        assert_vec4_approx_eq(a.col1, b.col1);
        assert_vec4_approx_eq(a.col2, b.col2);
        assert_vec4_approx_eq(a.col3, b.col3);
        assert_vec4_approx_eq(a.col4, b.col4);
    }

    #[test]
    fn test_inverse_of_combined_transform() {
        let rot_axis = Vec3::new(1.0, 2.0, 3.0);
        let m = Mat4::from_translation(Vec3::new(10.0, -5.0, 2.0))
            * Mat4::from_axis_angle(rot_axis, 1.57);
        let m_inv = m.inverse().expect("Matrix should be invertible");
        let identity = m * m_inv;
        assert_mat4_approx_eq(identity, Mat4::identity());
    }

    #[test]
    fn test_inverse_of_identity() {
        let m = Mat4::<f32>::identity();
        let inv = m.inverse().unwrap();
        assert_mat4_approx_eq(m, inv);
    }

    #[test]
    fn test_inverse_of_translation() {
        let t = Vec3::new(1.0, 2.0, 3.0);
        let m = Mat4::from_translation(t);
        let m_inv = m.inverse().unwrap();
        let expected_inv = Mat4::from_translation(-t);
        assert_mat4_approx_eq(m_inv, expected_inv);
    }

    #[test]
    fn test_inverse_of_scale() {
        let s = Vec3::new(2.0, 4.0, 0.5);
        let m = Mat4::from_scale(s);
        let m_inv = m.inverse().unwrap();
        let expected_inv = Mat4::from_scale(Vec3::new(1.0 / s.x, 1.0 / s.y, 1.0 / s.z));
        assert_mat4_approx_eq(m_inv, expected_inv);
    }

    #[test]
    fn test_inverse_of_singular_matrix() {
        let m = Mat4::from_scale(Vec3::new(1.0, 0.0, 1.0));
        let m_inv = m.inverse();
        assert!(m_inv.is_none());
    }

    #[test]
    fn test_multiplication_order() {
        let r = Mat4::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), std::f32::consts::FRAC_PI_2);
        let t = Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0));

        let rt = t * r;
        let tr = r * t;

        assert_ne!(rt, tr);

        let p = Vec4::new(1.0, 0.0, 0.0, 1.0);

        let p_rt = rt * p;
        assert_vec4_approx_eq(p_rt, Vec4::new(10.0, 0.0, -1.0, 1.0));

        let p_tr = tr * p;
        assert_vec4_approx_eq(p_tr, Vec4::new(0.0, 0.0, -11.0, 1.0));
    }

    #[test]
    fn test_identity_multiplication() {
        let m = Mat4::from_axis_angle(Vec3::new(0.3, 0.4, 0.5), 0.5);
        let i = Mat4::identity();
        assert_eq!(m * i, m);
        assert_eq!(i * m, m);
    }
}
