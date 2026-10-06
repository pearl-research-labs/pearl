#![allow(clippy::needless_range_loop)]

//! Each `vN` module is one proof scheme and may import only from older versions.
//! [`ffi`] is the binding layer shared by the Go and Python bindings: it may import
//! from every version, and no version imports it.

pub mod ffi;
pub mod v1;
pub mod v2;
pub mod v4;
pub mod v5;

/// Macro for checking equality with anyhow::Result error handling
/// Similar to assert_eq! but returns an error instead of panicking
#[macro_export]
macro_rules! ensure_eq {
    ($left:expr, $right:expr) => {
        if !($left == $right) {
            return Err(anyhow::anyhow!(
                "ensure_eq failed: left = {:?}, right = {:?}",
                $left,
                $right
            ));
        }
    };
    ($left:expr, $right:expr, $($arg:tt)+) => {
        if !($left == $right) {
            return Err(anyhow::anyhow!(
                "ensure_eq failed: {} || left = {:?}, right = {:?}",
                format!($($arg)+),
                $left,
                $right,
            ));
        }
    };
}

#[cfg(test)]
mod ensure_eq_tests {
    use anyhow::Result;

    #[test]
    fn test_ensure_eq_success() {
        fn check() -> Result<()> {
            ensure_eq!(5, 5);
            ensure_eq!("hello", "hello");
            ensure_eq!(vec![1, 2], vec![1, 2]);
            Ok(())
        }

        assert!(check().is_ok());
    }

    #[test]
    fn test_ensure_eq_failure() {
        fn check() -> Result<()> {
            ensure_eq!(5, 6);
            Ok(())
        }

        let result = check();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("left = 5, right = 6"));
    }

    #[test]
    fn test_ensure_eq_with_message() {
        fn check() -> Result<()> {
            ensure_eq!(5, 6, "Numbers should be equal");
            Ok(())
        }

        let result = check();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Numbers should be equal"));
    }
}
