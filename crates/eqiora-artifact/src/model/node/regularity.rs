//! Required authored Field regularity on the current wire.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum WireSpatialRegularity {
    Unspecified,
    L2,
    H1,
    HCurl,
    HDiv,
    Smooth,
}

impl WireSpatialRegularity {
    pub(super) fn encode(value: eqiora_schema::kernel::SpatialRegularity) -> Self {
        use eqiora_schema::kernel::SpatialRegularity as R;
        match value {
            R::Unspecified => Self::Unspecified,
            R::L2 => Self::L2,
            R::H1 => Self::H1,
            R::HCurl => Self::HCurl,
            R::HDiv => Self::HDiv,
            R::Smooth => Self::Smooth,
        }
    }

    pub(super) fn decode(self) -> eqiora_schema::kernel::SpatialRegularity {
        use eqiora_schema::kernel::SpatialRegularity as R;
        match self {
            Self::Unspecified => R::Unspecified,
            Self::L2 => R::L2,
            Self::H1 => R::H1,
            Self::HCurl => R::HCurl,
            Self::HDiv => R::HDiv,
            Self::Smooth => R::Smooth,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::WireNode;
    use eqiora_core::{DimExponents, Id, ScalarDomain, ValueFrame, ValueShape, ValueType};
    use eqiora_schema::kernel::{FieldDef, FieldRole, KernelNode, SpatialRegularity};

    #[test]
    fn all_regularities_are_required_and_roundtrip_exactly() {
        for regularity in [
            SpatialRegularity::Unspecified,
            SpatialRegularity::L2,
            SpatialRegularity::H1,
            SpatialRegularity::HCurl,
            SpatialRegularity::HDiv,
            SpatialRegularity::Smooth,
        ] {
            let value_type = ValueType::shaped(
                ScalarDomain::Real,
                DimExponents::DIMENSIONLESS,
                ValueShape::new([2]).unwrap(),
                ValueFrame::SpatialCartesian,
            )
            .unwrap();
            let node = KernelNode::from(
                FieldDef::new(Id::new(), value_type, FieldRole::Variable)
                    .with_spatial_regularity(regularity),
            );
            let wire = WireNode::encode(&node).unwrap();
            let bytes = serde_json::to_vec(&wire).unwrap();
            let replay: WireNode = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(replay.decode().unwrap(), node);
            let mut missing = serde_json::to_value(&wire).unwrap();
            assert!(
                missing["definition"]
                    .as_object_mut()
                    .unwrap()
                    .remove("spatial_regularity")
                    .is_some()
            );
            assert!(serde_json::from_value::<WireNode>(missing).is_err());
            let mut unknown = serde_json::to_value(&wire).unwrap();
            unknown["definition"]["spatial_regularity"] = "invented-space".into();
            assert!(serde_json::from_value::<WireNode>(unknown).is_err());
        }
    }
}
