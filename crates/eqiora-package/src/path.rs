use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use unicode_normalization::UnicodeNormalization;

use crate::ContractError;

/// A canonical, portability-constrained UTF-8 bundle path.
#[derive(Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NormalizedRelativePath(String);

impl NormalizedRelativePath {
    pub fn parse(value: impl Into<String>) -> Result<Self, ContractError> {
        let value = value.into();
        if value.is_empty() || value.len() > 4096 {
            return Err(ContractError::new(
                "bundle path must contain between 1 and 4096 UTF-8 bytes",
            ));
        }
        if value.contains('\0')
            || value.contains('\\')
            || value.starts_with('/')
            || value
                .chars()
                .any(|character| matches!(character, '<' | '>' | ':' | '"' | '|' | '?' | '*'))
        {
            return Err(ContractError::new(format!(
                "bundle path `{value}` is not a normalized relative path"
            )));
        }
        if value.split('/').any(|segment| {
            segment.is_empty()
                || segment == "."
                || segment == ".."
                || segment.starts_with(' ')
                || segment.ends_with([' ', '.'])
                || is_windows_device_name(segment)
        }) {
            return Err(ContractError::new(format!(
                "bundle path `{value}` contains a non-portable path segment"
            )));
        }
        if value.chars().any(char::is_control) || value.nfc().collect::<String>() != value {
            return Err(ContractError::new(format!(
                "bundle path `{value}` must be control-free NFC UTF-8"
            )));
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub(crate) fn ascii_case_key(&self) -> String {
        self.0.to_ascii_lowercase()
    }
}

fn is_windows_device_name(segment: &str) -> bool {
    let stem = segment.split('.').next().unwrap_or(segment).trim_end();
    let upper = stem.to_ascii_uppercase();
    if matches!(upper.as_str(), "CON" | "PRN" | "AUX" | "NUL") {
        return true;
    }
    let mut characters = upper.chars();
    let prefix: String = characters.by_ref().take(3).collect();
    let suffix: String = characters.collect();
    matches!(prefix.as_str(), "COM" | "LPT")
        && matches!(
            suffix.as_str(),
            "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "¹" | "²" | "³"
        )
}

impl fmt::Debug for NormalizedRelativePath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("NormalizedRelativePath")
            .field(&self.0)
            .finish()
    }
}

impl fmt::Display for NormalizedRelativePath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Serialize for NormalizedRelativePath {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for NormalizedRelativePath {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::parse(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unicode_18_combining_marks_require_canonical_order() {
        // UnicodeData 18.0.0 assigns U+1AEE CCC 220; U+0301 has CCC 230.
        // UAX #15 therefore orders 1AEE before 0301, with no composition
        // involving the digit zero. In Unicode 17, unassigned 1AEE had CCC 0.
        // https://www.unicode.org/Public/18.0.0/ucd/UnicodeData.txt
        let canonical = "0\u{1aee}\u{301}.eqi";
        let accepted = NormalizedRelativePath::parse(canonical).unwrap();
        assert_eq!(accepted.as_str(), canonical);

        let noncanonical = "0\u{301}\u{1aee}.eqi";
        let error = NormalizedRelativePath::parse(noncanonical).unwrap_err();
        assert!(error.to_string().contains("must be control-free NFC UTF-8"));
    }

    #[test]
    fn rejects_ambiguous_or_platform_dependent_paths() {
        assert!(NormalizedRelativePath::parse("models/resistor.eqi").is_ok());
        for path in [
            "",
            "/root.eqi",
            "./root.eqi",
            "a/../root.eqi",
            "a//b",
            "C:/root.eqi",
            "a\\b",
            "nul\0path",
            "cafe\u{301}.eqi",
            "stream:meta.eqi",
            "trailing.",
            "trailing ",
            " leading.eqi",
            "CON",
            "nul.txt",
            "src/Com1.eqi",
            "COM0",
            "CON .txt",
            "NUL .x",
            "LPT².log",
            "question?.eqi",
        ] {
            assert!(
                NormalizedRelativePath::parse(path).is_err(),
                "accepted {path:?}"
            );
        }
    }
}
