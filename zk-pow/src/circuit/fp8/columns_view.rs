//! Array conversions and column indices for homogeneous `#[repr(C)]` trace views.

use core::mem::{ManuallyDrop, transmute_copy};

/// Bit-reinterprets `T` as `U`. Unlike `transmute`, compiles for generic types (whose sizes the
/// compiler cannot prove equal); the equality is asserted at runtime in debug builds.
///
/// # Safety
/// `T` and `U` must have the same size, and every bit pattern of `T` must be valid at `U`.
pub(crate) unsafe fn transmute_no_compile_time_size_checks<T, U>(value: T) -> U {
    debug_assert_eq!(size_of::<T>(), size_of::<U>());
    // `ManuallyDrop` prevents a double drop: the returned `U` now owns the bits.
    let value = ManuallyDrop::new(value);
    unsafe { transmute_copy(&value) }
}

/// Implement array conversions, borrows and `Default` for `$n` fields of a
/// single `Copy` type. `$map` gives each field's flat trace index.
macro_rules! columns_view {
    ($view:ident, $n:ident, $map:ident) => {
        impl<T: Copy> From<[T; $n]> for $view<T> {
            fn from(value: [T; $n]) -> Self {
                unsafe { crate::circuit::fp8::columns_view::transmute_no_compile_time_size_checks(value) }
            }
        }

        impl<T: Copy> From<$view<T>> for [T; $n] {
            fn from(value: $view<T>) -> Self {
                unsafe { crate::circuit::fp8::columns_view::transmute_no_compile_time_size_checks(value) }
            }
        }

        impl<T: Copy> core::borrow::Borrow<$view<T>> for [T; $n] {
            fn borrow(&self) -> &$view<T> {
                unsafe { &*(self as *const [T; $n]).cast::<$view<T>>() }
            }
        }

        impl<T: Copy> core::borrow::BorrowMut<$view<T>> for [T; $n] {
            fn borrow_mut(&mut self) -> &mut $view<T> {
                unsafe { &mut *(self as *mut [T; $n]).cast::<$view<T>>() }
            }
        }

        impl<T: Copy> core::borrow::Borrow<[T; $n]> for $view<T> {
            fn borrow(&self) -> &[T; $n] {
                unsafe { &*(self as *const $view<T>).cast::<[T; $n]>() }
            }
        }

        impl<T: Copy> core::borrow::BorrowMut<[T; $n]> for $view<T> {
            fn borrow_mut(&mut self) -> &mut [T; $n] {
                unsafe { &mut *(self as *mut $view<T>).cast::<[T; $n]>() }
            }
        }

        impl<T: Copy + Default> Default for $view<T> {
            fn default() -> Self {
                [T::default(); $n].into()
            }
        }

        /// Maps each column name to its flat index in the trace row; use this to declare lookups
        /// and cross-table lookups over this STARK's columns.
        pub const $map: $view<usize> = {
            let mut indices = [0usize; $n];
            let mut i = 0;
            while i < $n {
                indices[i] = i;
                i += 1;
            }
            // SAFETY: the view is `repr(C)` over exactly `$n` `usize` fields, so it has the same
            // layout as `[usize; $n]`.
            unsafe { core::mem::transmute::<[usize; $n], $view<usize>>(indices) }
        };
    };
}

pub(crate) use columns_view;
