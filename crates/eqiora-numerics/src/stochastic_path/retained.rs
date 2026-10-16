//! Reuse the canonical bounded rational value codec for retained path arithmetic.
use eqiora_schema::kernel::pure_operator::ExactRational;
use num_rational::BigRational;
use serde::{Deserialize, Deserializer, Serialize, Serializer, ser::Error};

pub(super) mod value {
    use super::*;
    pub(crate) fn serialize<S: Serializer>(
        value: &BigRational,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        ExactRational::try_from(value.clone())
            .map_err(S::Error::custom)?
            .serialize(serializer)
    }
    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<BigRational, D::Error> {
        ExactRational::deserialize(deserializer).map(Into::into)
    }
}

pub(super) mod values {
    use super::*;
    pub(crate) fn serialize<S: Serializer>(
        values: &[BigRational],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let values = values
            .iter()
            .cloned()
            .map(ExactRational::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(S::Error::custom)?;
        values.serialize(serializer)
    }
    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vec<BigRational>, D::Error> {
        Vec::<ExactRational>::deserialize(deserializer)
            .map(|values| values.into_iter().map(Into::into).collect())
    }
}
