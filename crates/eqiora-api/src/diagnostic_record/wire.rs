//! Private wire adapters for the shared kernel diagnostic values.
use super::*;
use serde::{Deserializer, Serializer};

pub(super) mod severity {
    use super::*;
    #[derive(Serialize, Deserialize)]
    #[serde(rename_all = "kebab-case")]
    enum WireSeverity {
        Error,
        Warning,
        Note,
    }
    pub fn serialize<S: Serializer>(value: &Severity, serializer: S) -> Result<S::Ok, S::Error> {
        match value {
            Severity::Error => WireSeverity::Error,
            Severity::Warning => WireSeverity::Warning,
            Severity::Note => WireSeverity::Note,
        }
        .serialize(serializer)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Severity, D::Error> {
        Ok(match WireSeverity::deserialize(deserializer)? {
            WireSeverity::Error => Severity::Error,
            WireSeverity::Warning => Severity::Warning,
            WireSeverity::Note => Severity::Note,
        })
    }
}

pub(super) mod span {
    use super::*;
    #[derive(Serialize, Deserialize)]
    #[serde(remote = "Span", rename_all = "camelCase", deny_unknown_fields)]
    struct WireSpan {
        file: String,
        start: u32,
        end: u32,
    }
    #[derive(Serialize)]
    struct Borrowed<'a>(#[serde(with = "WireSpan")] &'a Span);
    #[derive(Deserialize)]
    struct Owned(#[serde(with = "WireSpan")] Span);
    pub fn serialize<S: Serializer>(
        value: &Option<Box<Span>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        value.as_deref().map(Borrowed).serialize(serializer)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Box<Span>>, D::Error> {
        Ok(Option::<Owned>::deserialize(deserializer)?.map(|value| Box::new(value.0)))
    }
}

pub(super) mod patch {
    use super::*;
    #[derive(Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct WirePatch<T> {
        summary: T,
    }
    pub fn serialize<S: Serializer>(
        value: &Option<Box<Patch>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        value
            .as_deref()
            .map(|patch| WirePatch {
                summary: patch.summary.as_str(),
            })
            .serialize(serializer)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Box<Patch>>, D::Error> {
        Ok(Option::<WirePatch<String>>::deserialize(deserializer)?
            .map(|value| Box::new(Patch::new(value.summary))))
    }
}
