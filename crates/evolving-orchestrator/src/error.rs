use thiserror::Error;

#[derive(Debug, Error)]
pub enum EvolveError {
    #[error("configuration error: {0}")]
    Config(String),

    #[error("design error: {0}")]
    Design(String),

    #[error("episode error: {0}")]
    Episode(String),

    #[error("judgment error: {0}")]
    Judgment(String),

    #[error("ranking error: {0}")]
    Ranking(String),

    #[error("training error: {0}")]
    Training(String),

    #[error("io error: {0}")]
    Io(String),

    #[error("json error: {0}")]
    Json(String),
}

pub type Result<T, E = EvolveError> = std::result::Result<T, E>;

impl From<std::io::Error> for EvolveError {
    fn from(value: std::io::Error) -> Self {
        EvolveError::Io(value.to_string())
    }
}

impl From<serde_json::Error> for EvolveError {
    fn from(value: serde_json::Error) -> Self {
        EvolveError::Json(value.to_string())
    }
}
