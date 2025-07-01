use num_traits::{Float, One, Zero};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

macro_rules! impl_vector {
    (
        $(#[$meta:meta])*
        $name:ident, ($($component:ident),+), $size:expr
    ) => {
        $(#[$meta])*
        #[repr(C)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name<T> {
            $(pub $component: T),+
        }

        impl<T: Copy> $name<T> {
            /// Creates a new vector from its components.
            ///
            /// # Examples
            ///
            /// ```
            /// # use math_library::Vec2;
            /// let v = Vec2::new(1.0f32, 2.0f32);
            /// assert_eq!(v.x, 1.0);
            /// assert_eq!(v.y, 2.0);
            /// ```
            #[inline]
            pub const fn new($($component: T),+) -> Self {
                Self { $($component),+ }
            }

            /// Creates a new vector with all components set to a single value.
            ///
            /// # Examples
            ///
            /// ```
            /// # use math_library::Vec3;
            /// let v = Vec3::splat(5.0f32);
            /// assert_eq!(v, Vec3::new(5.0, 5.0, 5.0));
            /// ```
            #[inline]
            pub const fn splat(value: T) -> Self {
                Self { $($component: value),+ }
            }
        }

        impl<T: Copy + Zero> Default for $name<T> {
            /// Returns a zero vector.
            ///
            /// # Examples
            ///
            /// ```
            /// # use math_library::Vec2;
            /// let v: Vec2<f32> = Default::default();
            /// assert_eq!(v, Vec2::new(0.0, 0.0));
            /// ```
            #[inline]
            fn default() -> Self {
                Self::splat(T::zero())
            }
        }

        impl<T: Copy> From<[T; $size]> for $name<T> {
            /// Creates a vector from an array.
            ///
            /// # Examples
            ///
            /// ```
            /// # use math_library::Vec3;
            /// let arr = [1i32, 2, 3];
            /// let v = Vec3::from(arr);
            /// assert_eq!(v.x, 1);
            /// ```
            #[inline]
            fn from(arr: [T; $size]) -> Self {
                let [$($component),+] = arr;
                Self { $($component),+ }
            }
        }

        impl<T: Copy> From<$name<T>> for [T; $size] {
            /// Creates an array from a vector.
            ///
            /// # Examples
            ///
            /// ```
            /// # use math_library::Vec3;
            /// let v = Vec3::new(1i32, 2, 3);
            /// let arr: [i32; 3] = v.into();
            /// assert_eq!(arr, [1, 2, 3]);
            /// ```
            #[inline]
            fn from(v: $name<T>) -> Self {
                [$(v.$component),+]
            }
        }

        impl<T: Copy + Neg<Output = T>> Neg for $name<T> {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self::Output {
                Self { $($component: -self.$component),+ }
            }
        }

        impl<T: Copy + Add<Output = T>> Add for $name<T> {
            type Output = Self;
            #[inline]
            fn add(self, rhs: Self) -> Self {
                Self { $($component: self.$component + rhs.$component),+ }
            }
        }

        impl<T: Copy + AddAssign> AddAssign for $name<T> {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                $(self.$component += rhs.$component;)+
            }
        }

        impl<T: Copy + Sub<Output = T>> Sub for $name<T> {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: Self) -> Self {
                Self { $($component: self.$component - rhs.$component),+ }
            }
        }

        impl<T: Copy + SubAssign> SubAssign for $name<T> {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                $(self.$component -= rhs.$component;)+
            }
        }

        impl<T: Copy + Mul<Output = T>> Mul for $name<T> {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: Self) -> Self {
                Self { $($component: self.$component * rhs.$component),+ }
            }
        }

        impl<T: Copy + Mul<Output = T>> Mul<T> for $name<T> {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: T) -> Self {
                Self { $($component: self.$component * rhs),+ }
            }
        }

        impl<T: Copy + MulAssign> MulAssign<T> for $name<T> {
            #[inline]
            fn mul_assign(&mut self, rhs: T) {
                $(self.$component *= rhs;)+
            }
        }

        impl<T: Copy + Div<Output = T> + Zero + PartialEq> Div<T> for $name<T> {
            type Output = Self;
            #[inline]
            fn div(self, rhs: T) -> Self {
                if rhs.is_zero() {
                    panic!("attempt to divide vector by zero");
                }
                Self { $($component: self.$component / rhs),+ }
            }
        }

        impl<T: Copy + DivAssign<T> + Zero + PartialEq> DivAssign<T> for $name<T> {
            #[inline]
            fn div_assign(&mut self, rhs: T) {
                if rhs.is_zero() {
                    panic!("attempt to divide vector by zero");
                }
                $(self.$component /= rhs;)+
            }
        }

        impl<T> $name<T>
        where
            T: Copy + Mul<Output = T> + Add<Output = T> + Zero,
        {
            /// Calculates the dot product of two vectors.
            ///
            /// # Examples
            ///
            /// ```
            /// # use math_library::Vec3;
            /// let v1 = Vec3::new(1.0f32, 2.0, 3.0);
            /// let v2 = Vec3::new(4.0, 5.0, 6.0);
            /// assert_eq!(v1.dot(v2), 32.0);
            /// ```
            #[inline]
            pub fn dot(self, other: Self) -> T {
                let prod = self * other;
                T::zero() $(+ prod.$component)+
            }
        }

        impl<T> $name<T>
        where
            T: Float,
        {
            /// Calculates the squared length of the vector.
            ///
            /// This is faster than `length()` as it avoids a square root.
            /// It is often used for comparing vector lengths.
            ///
            /// # Examples
            ///
            /// ```
            /// # use math_library::Vec2;
            /// let v = Vec2::new(3.0f32, 4.0f32);
            /// assert_eq!(v.length_squared(), 25.0);
            /// ```
            #[inline]
            pub fn length_squared(self) -> T {
                self.dot(self)
            }

            /// Calculates the length (magnitude) of the vector.
            ///
            /// # Examples
            ///
            /// ```
            /// # use math_library::Vec2;
            /// let v = Vec2::new(3.0f32, 4.0f32);
            /// assert_eq!(v.length(), 5.0);
            /// ```
            #[inline]
            pub fn length(self) -> T {
                self.length_squared().sqrt()
            }

            /// Returns a normalized version of the vector (with a length of 1).
            ///
            /// Returns a zero vector if the original vector has a length of zero.
            ///
            /// # Examples
            ///
            /// ```
            /// # use math_library::Vec2;
            /// let v: Vec2<f32> = Vec2::new(3.0, 4.0);
            /// let norm = v.normalize();
            /// assert!((norm.length() - 1.0).abs() < 1e-6);
            /// ```
            #[inline]
            pub fn normalize(self) -> Self {
                let len = self.length();
                if len <= T::epsilon() {
                    Self::default()
                } else {
                    self / len
                }
            }
        }
    };
}

impl_vector!(
    /// A 2-dimensional vector.
    Vec2,
    (x, y),
    2
);
impl_vector!(
    /// A 3-dimensional vector.
    Vec3,
    (x, y, z),
    3
);
impl_vector!(
    /// A 4-dimensional vector.
    Vec4,
    (x, y, z, w),
    4
);

impl<T> Vec3<T>
where
    T: Copy + Mul<Output = T> + Sub<Output = T>,
{
    /// Calculates the cross product of two vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// # use math_library::Vec3;
    /// let i = Vec3::new(1.0f32, 0.0, 0.0);
    /// let j = Vec3::new(0.0, 1.0, 0.0);
    /// let k = Vec3::new(0.0, 0.0, 1.0);
    /// assert_eq!(i.cross(j), k);
    /// ```
    #[inline]
    pub fn cross(self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }
}

impl<T: Copy + Zero + One> Vec3<T> {
    /// Converts a `Vec3` to a `Vec4` representing a point in homogeneous coordinates.
    ///
    /// The `w` component is set to `1`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use math_library::{Vec3, Vec4};
    /// let v3 = Vec3::new(1i32, 2, 3);
    /// let v4 = v3.to_vec4_point();
    /// assert_eq!(v4, Vec4::new(1, 2, 3, 1));
    /// ```
    #[inline]
    pub fn to_vec4_point(self) -> Vec4<T> {
        Vec4::new(self.x, self.y, self.z, T::one())
    }

    /// Converts a `Vec3` to a `Vec4` representing a direction in homogeneous coordinates.
    ///
    /// The `w` component is set to `0`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use math_library::{Vec3, Vec4};
    /// let v3 = Vec3::new(1i32, 2, 3);
    /// let v4 = v3.to_vec4_vector();
    /// assert_eq!(v4, Vec4::new(1, 2, 3, 0));
    /// ```
    #[inline]
    pub fn to_vec4_vector(self) -> Vec4<T> {
        Vec4::new(self.x, self.y, self.z, T::zero())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const EPSILON: f64 = 1e-9;

    #[test]
    fn test_ops() {
        let v1 = Vec2::new(1.0, 2.0);
        let v2 = Vec2::new(3.0, 4.0);
        assert_eq!(v1 + v2, Vec2::new(4.0, 6.0));
        assert_eq!(v2 - v1, Vec2::new(2.0, 2.0));
        assert_eq!(v1 * 2.0, Vec2::new(2.0, 4.0));
        assert_eq!(-v1, Vec2::new(-1.0, -2.0));
    }

    #[test]
    fn test_dot_product() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(4.0, -5.0, 6.0);
        assert!((v1.dot(v2) - 12.0).abs() < EPSILON);
        assert!((v1.dot(Vec3::default()) - 0.0).abs() < EPSILON);
        let i = Vec3::new(1.0, 0.0, 0.0);
        let j = Vec3::new(0.0, 1.0, 0.0);
        assert!((i.dot(j) - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_length_and_normalize() {
        let v1 = Vec4::new(3.0, 4.0, 0.0, 0.0);
        assert!((v1.length() - 5.0).abs() < EPSILON);
        let norm = v1.normalize();
        assert!((norm.x - 0.6).abs() < EPSILON);
        assert!((norm.y - 0.8).abs() < EPSILON);
        assert!((norm.length() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let zero = Vec3::<f64>::default();
        let normalized_zero = zero.normalize();
        assert_eq!(normalized_zero, zero);
    }

    #[test]
    fn test_cross_product() {
        let i = Vec3::new(1.0, 0.0, 0.0);
        let j = Vec3::new(0.0, 1.0, 0.0);
        let k = Vec3::new(0.0, 0.0, 1.0);
        assert_eq!(i.cross(j), k);
        assert_eq!(j.cross(i), -k);
        assert_eq!(i.cross(i), Vec3::default());
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v.cross(v * 2.0), Vec3::default());
    }

    #[test]
    #[should_panic]
    fn test_division_by_zero() {
        let v = Vec2::new(1.0, 1.0);
        let _ = v / 0.0;
    }
}
