//! Complete-exterior proofs over exact occurrence identities.
//!
//! Cartesian source domains retain the axis/side bijection. Geometry selections
//! delegate coverage and incidence to their authenticated Geometry owner.

use eqiora_schema::kernel::BoundarySide;

/// Domain evidence available to the complete-exterior proof.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum ExteriorDomain<I> {
    GeometryVolume {
        geometry: std::sync::Arc<eqiora_geometry::CanonicalGeometryV1>,
        selection: String,
        ambient_dimension: usize,
    },
    GeometryBoundary {
        geometry: std::sync::Arc<eqiora_geometry::CanonicalGeometryV1>,
        selection: String,
        parent_selection: String,
        exact_parent: I,
        ambient_dimension: usize,
        embedding: Option<eqiora_schema::kernel::CartesianBoundaryEmbedding>,
    },
    /// Exact Cartesian volume identity and its authoritative dimension.
    Volume { ambient_dimension: usize },
    /// Exact Cartesian boundary identity metadata.
    Boundary {
        exact_parent: I,
        ambient_dimension: usize,
        axis: usize,
        side: BoundarySide,
    },
}

/// An explicit, nonempty set of sorted unique exact Boundary identities.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct BoundarySet<I> {
    exact_parent: I,
    members: Box<[I]>,
}

impl<I> BoundarySet<I>
where
    I: Clone + Ord,
{
    pub(super) fn try_new(
        exact_parent: I,
        members: impl IntoIterator<Item = I>,
    ) -> Result<Self, CompleteExteriorError<I>> {
        let mut canonical_members = Vec::new();
        for member in members {
            if canonical_members.len() == canonical_members.capacity() {
                let count = canonical_members.len().checked_add(1).ok_or(
                    CompleteExteriorError::Allocation {
                        resource: "exact Boundary identities",
                        count: usize::MAX,
                    },
                )?;
                canonical_members.try_reserve(1).map_err(|_| {
                    CompleteExteriorError::Allocation {
                        resource: "exact Boundary identities",
                        count,
                    }
                })?;
            }
            canonical_members.push(member);
        }
        let mut members = canonical_members;
        if members.is_empty() {
            return Err(CompleteExteriorError::Empty);
        }
        members.sort_unstable();
        if let Some(pair) = members.windows(2).find(|pair| pair[0] == pair[1]) {
            return Err(CompleteExteriorError::DuplicateExactMember {
                member: pair[0].clone(),
            });
        }
        Ok(Self {
            exact_parent,
            members: members.into_boxed_slice(),
        })
    }

    /// Exact parent supplied by the occurrence's volume-support binding.
    #[cfg(test)]
    pub(super) const fn exact_parent(&self) -> &I {
        &self.exact_parent
    }

    /// Members in canonical exact-identity order, never author list order.
    #[cfg(test)]
    pub(super) const fn members(&self) -> &[I] {
        &self.members
    }
}

/// One exact Boundary member with optional genuine Cartesian metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct CompleteExteriorMember<I> {
    cartesian: Option<(usize, BoundarySide)>,
    boundary: I,
    selection: Option<String>,
}

impl<I> CompleteExteriorMember<I> {
    pub(super) const fn cartesian(&self) -> Option<(usize, BoundarySide)> {
        self.cartesian
    }

    pub(super) fn selection(&self) -> Option<&str> {
        self.selection.as_deref()
    }

    pub(super) const fn boundary(&self) -> &I {
        &self.boundary
    }
}

/// Successful complete partition with exact member identities.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct CompleteExteriorWitness<I> {
    exact_parent: I,
    ambient_dimension: usize,
    members: Box<[CompleteExteriorMember<I>]>,
}

impl<I> CompleteExteriorWitness<I> {
    pub(super) const fn exact_parent(&self) -> &I {
        &self.exact_parent
    }

    pub(super) const fn ambient_dimension(&self) -> usize {
        self.ambient_dimension
    }

    pub(super) const fn members(&self) -> &[CompleteExteriorMember<I>] {
        &self.members
    }
}

/// Closed failures of complete-exterior proof.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum CompleteExteriorError<I> {
    Empty,
    GeometryPartition {
        message: String,
    },
    DuplicateExactMember {
        member: I,
    },
    UnknownDomain {
        domain: I,
    },
    ExactParentIsNotVolume {
        exact_parent: I,
    },
    ExactParentHasZeroDimension {
        exact_parent: I,
    },
    MemberIsVolume {
        member: I,
    },
    BoundaryOfBoundary {
        member: I,
        actual_parent: I,
    },
    WrongParent {
        member: I,
        expected_parent: I,
        actual_parent: I,
    },
    WrongDimension {
        member: I,
        expected: usize,
        actual: usize,
    },
    AxisOutsideParentDimension {
        member: I,
        axis: usize,
        ambient_dimension: usize,
    },
    DuplicateGeometry {
        axis: usize,
        side: BoundarySide,
        first: I,
        duplicate: I,
    },
    MissingSide {
        axis: usize,
        side: BoundarySide,
    },
    SideCountOverflow {
        ambient_dimension: usize,
    },
    Allocation {
        resource: &'static str,
        count: usize,
    },
}

/// Prove an explicit Boundary set is exactly the exterior of one Cartesian
/// volume.
///
/// The parent dimension is read once from `resolve(exact_parent)`. Boundary
/// declaration order and lexical side names never participate. Validation is
/// atomic: no partial witness is returned for any missing, repeated, or
/// ill-typed member.
pub(super) fn prove_complete_exterior<I>(
    exact_parent: I,
    members: impl IntoIterator<Item = I>,
    mut resolve: impl FnMut(&I) -> Option<ExteriorDomain<I>>,
) -> Result<CompleteExteriorWitness<I>, CompleteExteriorError<I>>
where
    I: Clone + Ord,
{
    let set = BoundarySet::try_new(exact_parent, members)?;
    prove_boundary_set(set, &mut resolve)
}

fn prove_boundary_set<I>(
    set: BoundarySet<I>,
    resolve: &mut impl FnMut(&I) -> Option<ExteriorDomain<I>>,
) -> Result<CompleteExteriorWitness<I>, CompleteExteriorError<I>>
where
    I: Clone + Ord,
{
    let parent =
        resolve(&set.exact_parent).ok_or_else(|| CompleteExteriorError::UnknownDomain {
            domain: set.exact_parent.clone(),
        })?;
    if let ExteriorDomain::GeometryVolume {
        geometry,
        selection,
        ambient_dimension,
    } = parent
    {
        return prove_geometry_set(set, geometry, selection, ambient_dimension, resolve);
    }
    let ExteriorDomain::Volume { ambient_dimension } = parent else {
        return Err(CompleteExteriorError::ExactParentIsNotVolume {
            exact_parent: set.exact_parent,
        });
    };
    if ambient_dimension == 0 {
        return Err(CompleteExteriorError::ExactParentHasZeroDimension {
            exact_parent: set.exact_parent,
        });
    }
    let side_count = ambient_dimension
        .checked_mul(2)
        .ok_or(CompleteExteriorError::SideCountOverflow { ambient_dimension })?;
    let mut sides = Vec::new();
    sides
        .try_reserve_exact(side_count)
        .map_err(|_| CompleteExteriorError::Allocation {
            resource: "Cartesian exterior side slots",
            count: side_count,
        })?;
    sides.resize_with(side_count, || None);

    for member in &set.members {
        let domain = resolve(member).ok_or_else(|| CompleteExteriorError::UnknownDomain {
            domain: member.clone(),
        })?;
        let ExteriorDomain::Boundary {
            exact_parent: actual_parent,
            ambient_dimension: actual_dimension,
            axis,
            side,
        } = domain
        else {
            return Err(CompleteExteriorError::MemberIsVolume {
                member: member.clone(),
            });
        };

        if actual_parent != set.exact_parent {
            match resolve(&actual_parent) {
                Some(ExteriorDomain::Boundary { .. } | ExteriorDomain::GeometryBoundary { .. }) => {
                    return Err(CompleteExteriorError::BoundaryOfBoundary {
                        member: member.clone(),
                        actual_parent,
                    });
                }
                Some(ExteriorDomain::Volume { .. } | ExteriorDomain::GeometryVolume { .. }) => {
                    return Err(CompleteExteriorError::WrongParent {
                        member: member.clone(),
                        expected_parent: set.exact_parent.clone(),
                        actual_parent,
                    });
                }
                None => {
                    return Err(CompleteExteriorError::UnknownDomain {
                        domain: actual_parent,
                    });
                }
            }
        }
        if actual_dimension != ambient_dimension {
            return Err(CompleteExteriorError::WrongDimension {
                member: member.clone(),
                expected: ambient_dimension,
                actual: actual_dimension,
            });
        }
        if axis >= ambient_dimension {
            return Err(CompleteExteriorError::AxisOutsideParentDimension {
                member: member.clone(),
                axis,
                ambient_dimension,
            });
        }
        let side_offset = match side {
            BoundarySide::Lower => 0,
            BoundarySide::Upper => 1,
        };
        let index = axis
            .checked_mul(2)
            .and_then(|value| value.checked_add(side_offset))
            .ok_or(CompleteExteriorError::SideCountOverflow { ambient_dimension })?;
        if let Some(first) = sides[index].replace(member.clone()) {
            return Err(CompleteExteriorError::DuplicateGeometry {
                axis,
                side,
                first,
                duplicate: member.clone(),
            });
        }
    }

    let mut canonical = Vec::new();
    canonical
        .try_reserve_exact(side_count)
        .map_err(|_| CompleteExteriorError::Allocation {
            resource: "complete-exterior witness sides",
            count: side_count,
        })?;
    for axis in 0..ambient_dimension {
        for side in [BoundarySide::Lower, BoundarySide::Upper] {
            let side_offset = usize::from(matches!(side, BoundarySide::Upper));
            let index = axis * 2 + side_offset;
            let boundary = sides[index]
                .take()
                .ok_or(CompleteExteriorError::MissingSide { axis, side })?;
            canonical.push(CompleteExteriorMember {
                cartesian: Some((axis, side)),
                boundary,
                selection: None,
            });
        }
    }

    Ok(CompleteExteriorWitness {
        exact_parent: set.exact_parent,
        ambient_dimension,
        members: canonical.into_boxed_slice(),
    })
}

fn prove_geometry_set<I: Clone + Ord>(
    set: BoundarySet<I>,
    geometry: std::sync::Arc<eqiora_geometry::CanonicalGeometryV1>,
    parent_name: String,
    ambient_dimension: usize,
    resolve: &mut impl FnMut(&I) -> Option<ExteriorDomain<I>>,
) -> Result<CompleteExteriorWitness<I>, CompleteExteriorError<I>> {
    let invalid = |message: &str| CompleteExteriorError::GeometryPartition {
        message: message.to_owned(),
    };
    let parent = geometry
        .entity_set(&parent_name)
        .ok_or_else(|| invalid("missing Geometry parent selection"))?;
    let mut members = Vec::new();
    for identity in &set.members {
        let Some(ExteriorDomain::GeometryBoundary {
            geometry: owner,
            selection,
            parent_selection,
            exact_parent,
            ambient_dimension: dimensions,
            embedding,
        }) = resolve(identity)
        else {
            return Err(invalid(
                "exterior member has no authenticated Geometry selection",
            ));
        };
        if !std::sync::Arc::ptr_eq(&owner, &geometry)
            || parent_selection != parent_name
            || exact_parent != set.exact_parent
            || dimensions != ambient_dimension
        {
            return Err(invalid(
                "exterior member differs in Geometry owner, exact occurrence parent, selection parent or dimension",
            ));
        }
        members.push(CompleteExteriorMember {
            cartesian: embedding.map(|value| (value.normal_axis(), value.side())),
            boundary: identity.clone(),
            selection: Some(selection),
        });
    }
    let parts = members
        .iter()
        .map(|member| {
            geometry
                .entity_set(
                    member
                        .selection
                        .as_deref()
                        .expect("Geometry member selection"),
                )
                .ok_or_else(|| invalid("missing Geometry boundary selection"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    geometry
        .validate_complete_exterior_partition(
            parent,
            &parts,
            eqiora_geometry::CanonicalGeometryLimits::default(),
        )
        .map_err(|error| invalid(error.message()))?;
    // Cartesian single-side names/order stay stable. General parts use exact identity order.
    if members.iter().all(|member| member.cartesian.is_some()) {
        members.sort_by_key(|member| {
            member
                .cartesian
                .map(|(axis, side)| (axis, matches!(side, BoundarySide::Upper)))
        });
    }
    Ok(CompleteExteriorWitness {
        exact_parent: set.exact_parent,
        ambient_dimension,
        members: members.into_boxed_slice(),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    type Identity = &'static str;

    fn volume(ambient_dimension: usize) -> ExteriorDomain<Identity> {
        ExteriorDomain::Volume { ambient_dimension }
    }

    fn boundary(
        exact_parent: Identity,
        ambient_dimension: usize,
        axis: usize,
        side: BoundarySide,
    ) -> ExteriorDomain<Identity> {
        ExteriorDomain::Boundary {
            exact_parent,
            ambient_dimension,
            axis,
            side,
        }
    }

    fn square() -> BTreeMap<Identity, ExteriorDomain<Identity>> {
        BTreeMap::from([
            ("body", volume(2)),
            ("x_lower", boundary("body", 2, 0, BoundarySide::Lower)),
            ("x_upper", boundary("body", 2, 0, BoundarySide::Upper)),
            ("y_lower", boundary("body", 2, 1, BoundarySide::Lower)),
            ("y_upper", boundary("body", 2, 1, BoundarySide::Upper)),
        ])
    }

    fn prove(
        domains: &BTreeMap<Identity, ExteriorDomain<Identity>>,
        members: impl IntoIterator<Item = Identity>,
    ) -> Result<CompleteExteriorWitness<Identity>, CompleteExteriorError<Identity>> {
        prove_complete_exterior("body", members, |identity| domains.get(identity).cloned())
    }

    #[test]
    fn derives_dimension_and_canonical_bijection_from_exact_parent() {
        let mut domains = square();
        domains.insert(
            "unselected_duplicate",
            boundary("body", 2, 0, BoundarySide::Lower),
        );
        let witness = prove(&domains, ["y_upper", "x_lower", "y_lower", "x_upper"])
            .expect("unordered complete exterior is valid");

        assert_eq!(witness.exact_parent(), &"body");
        assert_eq!(witness.ambient_dimension(), 2);
        assert_eq!(
            witness
                .members()
                .iter()
                .map(|side| (
                    side.cartesian().unwrap().0,
                    side.cartesian().unwrap().1,
                    *side.boundary()
                ))
                .collect::<Vec<_>>(),
            [
                (0, BoundarySide::Lower, "x_lower"),
                (0, BoundarySide::Upper, "x_upper"),
                (1, BoundarySide::Lower, "y_lower"),
                (1, BoundarySide::Upper, "y_upper"),
            ]
        );

        let set = BoundarySet::try_new("body", ["y_upper", "x_lower", "y_lower", "x_upper"])
            .expect("complete input has a valid exact set");
        assert_eq!(set.exact_parent(), &"body");
        assert_eq!(set.members(), ["x_lower", "x_upper", "y_lower", "y_upper"]);
    }

    #[test]
    fn empty_and_duplicate_exact_members_fail_before_geometry() {
        let domains = square();
        assert_eq!(prove(&domains, []), Err(CompleteExteriorError::Empty));
        assert_eq!(
            prove(&domains, ["x_lower", "x_upper", "y_lower", "x_lower"]),
            Err(CompleteExteriorError::DuplicateExactMember { member: "x_lower" })
        );
    }

    #[test]
    fn missing_side_and_distinct_duplicate_geometry_are_rejected() {
        let mut domains = square();
        domains.insert(
            "another_x_lower",
            boundary("body", 2, 0, BoundarySide::Lower),
        );
        assert_eq!(
            prove(&domains, ["x_lower", "x_upper", "y_lower"]),
            Err(CompleteExteriorError::MissingSide {
                axis: 1,
                side: BoundarySide::Upper,
            })
        );
        assert_eq!(
            prove(
                &domains,
                [
                    "x_lower",
                    "another_x_lower",
                    "x_upper",
                    "y_lower",
                    "y_upper",
                ]
            ),
            Err(CompleteExteriorError::DuplicateGeometry {
                axis: 0,
                side: BoundarySide::Lower,
                first: "another_x_lower",
                duplicate: "x_lower",
            })
        );
    }

    #[test]
    fn wrong_parent_volume_member_and_boundary_of_boundary_are_distinct() {
        let mut domains = square();
        domains.insert("other", volume(2));
        domains.insert("other_side", boundary("other", 2, 1, BoundarySide::Upper));
        domains.insert("edge", boundary("x_lower", 2, 1, BoundarySide::Lower));

        assert_eq!(
            prove(&domains, ["x_lower", "x_upper", "y_lower", "other_side"]),
            Err(CompleteExteriorError::WrongParent {
                member: "other_side",
                expected_parent: "body",
                actual_parent: "other",
            })
        );
        assert_eq!(
            prove(&domains, ["body", "x_upper", "y_lower", "y_upper"]),
            Err(CompleteExteriorError::MemberIsVolume { member: "body" })
        );
        assert_eq!(
            prove(&domains, ["x_lower", "x_upper", "y_upper", "edge"]),
            Err(CompleteExteriorError::BoundaryOfBoundary {
                member: "edge",
                actual_parent: "x_lower",
            })
        );
    }

    #[test]
    fn wrong_dimension_and_invalid_axis_fail_closed() {
        let mut domains = square();
        domains.insert(
            "wrong_dimension",
            boundary("body", 3, 1, BoundarySide::Upper),
        );
        domains.insert("wrong_axis", boundary("body", 2, 2, BoundarySide::Upper));
        assert_eq!(
            prove(
                &domains,
                ["x_lower", "x_upper", "y_lower", "wrong_dimension"]
            ),
            Err(CompleteExteriorError::WrongDimension {
                member: "wrong_dimension",
                expected: 2,
                actual: 3,
            })
        );
        assert_eq!(
            prove(&domains, ["x_lower", "x_upper", "y_lower", "wrong_axis"]),
            Err(CompleteExteriorError::AxisOutsideParentDimension {
                member: "wrong_axis",
                axis: 2,
                ambient_dimension: 2,
            })
        );
    }

    #[test]
    fn exact_parent_shape_and_side_count_arithmetic_fail_closed() {
        let boundary_parent = BTreeMap::from([
            ("body", boundary("outer", 2, 0, BoundarySide::Lower)),
            ("outer", volume(2)),
        ]);
        assert_eq!(
            prove_complete_exterior("body", ["body"], |identity| {
                boundary_parent.get(identity).cloned()
            }),
            Err(CompleteExteriorError::ExactParentIsNotVolume {
                exact_parent: "body",
            })
        );

        let zero = BTreeMap::from([("body", volume(0))]);
        assert_eq!(
            prove(&zero, ["member"]),
            Err(CompleteExteriorError::ExactParentHasZeroDimension {
                exact_parent: "body",
            })
        );

        let overflow = BTreeMap::from([("body", volume(usize::MAX))]);
        assert_eq!(
            prove(&overflow, ["member"]),
            Err(CompleteExteriorError::SideCountOverflow {
                ambient_dimension: usize::MAX,
            })
        );
    }
}
