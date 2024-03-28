/// Main library error type.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// Arbitrary errors wrapping.
    #[error(transparent)]
    Wrapped(Box<dyn std::error::Error + Send + Sync>),

    /// User generated error message, typically created via `bail!`.
    #[error("{0}")]
    Msg(String),
}

impl Error {
    pub fn wrap(err: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Wrapped(Box::new(err))
    }

    pub fn msg(err: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Msg(err.to_string())
    }
}

#[macro_export]
macro_rules! bail {
    ($msg:literal $(,)?) => {
        return Err($crate::error::Error::Msg(format!($msg).into()))
    };
    ($err:expr $(,)?) => {
        return Err($crate::error::Error::Msg(format!($err).into()))
    };
    ($fmt:expr, $($arg:tt)*) => {
        return Err($crate::error::Error::Msg(format!($fmt, $($arg)*).into()))
    };
}

pub type Result<T> = std::result::Result<T, Error>;
