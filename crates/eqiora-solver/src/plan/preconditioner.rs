use super::*;
use crate::AlgebraicBlock;

/// Capability family; complete configuration remains in `PreconditionerPolicy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PreconditionerKind {
    /// Identity action.
    Identity,
    /// Inverse diagonal action.
    Jacobi,
    /// Fixed upper triangular diagonal-Schur action.
    UpperDiagonalSchur,
}

/// Exact unresolved provider choice for a fixed block factorization.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "WireFactorization", into = "WireFactorization")]
pub struct FixedFactorizationChoice {
    algorithm: LinearSolver,
    provider: String,
}
impl FixedFactorizationChoice {
    /// Select a fixed sparse LU factorization by provider source name.
    /// Provider release resolution belongs to the existing execution registry.
    ///
    /// # Errors
    /// Rejects non-SparseLU algorithms and empty or whitespace-padded providers.
    pub fn new(algorithm: LinearSolver, provider: impl Into<String>) -> Result<Self, Diagnostic> {
        let provider = provider.into();
        if algorithm != LinearSolver::SparseLu || provider.is_empty() || provider.trim() != provider {
            return Err(invalid_plan("fixed block factorization requires SparseLU and a nonempty provider source name"));
        }
        Ok(Self { algorithm, provider })
    }
    /// Fixed algorithm.
    pub const fn algorithm(&self) -> LinearSolver { self.algorithm }
    /// Unresolved provider source name.
    pub fn provider(&self) -> &str { &self.provider }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireFactorization { algorithm: LinearSolver, provider: String }
impl From<FixedFactorizationChoice> for WireFactorization {
    fn from(value: FixedFactorizationChoice) -> Self { Self {algorithm:value.algorithm, provider:value.provider} }
}
impl TryFrom<WireFactorization> for FixedFactorizationChoice {
    type Error=Diagnostic;
    fn try_from(value:WireFactorization)->Result<Self,Diagnostic> { Self::new(value.algorithm,value.provider) }
}

/// Complete policy of a fixed upper triangular diagonal-Schur composition.
/// Limits bound Schur construction entries and scalar products, not factor memory or RSS.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "WireUpperDiagonalSchur", into = "WireUpperDiagonalSchur")]
pub struct UpperDiagonalSchurConfig {
    groups: [Vec<AlgebraicBlock>; 2],
    a_factorization: FixedFactorizationChoice,
    schur_factorization: FixedFactorizationChoice,
    maximum_schur_entries: NonZeroUsize,
    maximum_schur_products: NonZeroUsize,
}
impl UpperDiagonalSchurConfig {
    /// Select exactly two ordered semantic groups and their fixed factor providers.
    /// Coordinate coverage and sharing require the retained realized structure at admission.
    ///
    /// # Errors
    /// Rejects empty groups, repeated semantic blocks, and zero construction limits.
    pub fn new(groups:[Vec<AlgebraicBlock>;2], a_factorization:FixedFactorizationChoice, schur_factorization:FixedFactorizationChoice, maximum_schur_entries:usize, maximum_schur_products:usize) -> Result<Self,Diagnostic> {
        let mut seen=std::collections::HashSet::new();
        if groups.iter().any(Vec::is_empty) || groups.iter().flatten().any(|block| !seen.insert(*block)) {
            return Err(invalid_plan("upper diagonal-Schur requires two nonempty groups with unique semantic blocks"));
        }
        let maximum_schur_entries=NonZeroUsize::new(maximum_schur_entries).ok_or_else(||invalid_plan("Schur entry limit must be nonzero"))?;
        let maximum_schur_products=NonZeroUsize::new(maximum_schur_products).ok_or_else(||invalid_plan("Schur product limit must be nonzero"))?;
        Ok(Self {groups,a_factorization,schur_factorization,maximum_schur_entries,maximum_schur_products})
    }
    /// Ordered exact semantic groups.
    pub const fn groups(&self)->&[Vec<AlgebraicBlock>;2] { &self.groups }
    /// Fixed A-block factor choice.
    pub const fn a_factorization(&self)->&FixedFactorizationChoice { &self.a_factorization }
    /// Fixed Schur-block factor choice.
    pub const fn schur_factorization(&self)->&FixedFactorizationChoice { &self.schur_factorization }
    /// Maximum retained Schur entries (not a factor-storage bound).
    pub const fn maximum_schur_entries(&self)->NonZeroUsize { self.maximum_schur_entries }
    /// Maximum Schur product contributions.
    pub const fn maximum_schur_products(&self)->NonZeroUsize { self.maximum_schur_products }
}
#[derive(Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
struct WireUpperDiagonalSchur {
    groups:[Vec<WireBlock>;2], a_factorization:FixedFactorizationChoice,
    schur_factorization:FixedFactorizationChoice, maximum_schur_entries:usize, maximum_schur_products:usize,
}
impl From<UpperDiagonalSchurConfig> for WireUpperDiagonalSchur {
    fn from(value:UpperDiagonalSchurConfig)->Self { Self {groups:value.groups.map(|group| group.into_iter().map(WireBlock::from).collect()),a_factorization:value.a_factorization,schur_factorization:value.schur_factorization,maximum_schur_entries:value.maximum_schur_entries.get(),maximum_schur_products:value.maximum_schur_products.get()} }
}
impl TryFrom<WireUpperDiagonalSchur> for UpperDiagonalSchurConfig {
    type Error=Diagnostic;
    fn try_from(value:WireUpperDiagonalSchur)->Result<Self,Diagnostic> { Self::new([value.groups[0].iter().map(AlgebraicBlock::try_from).collect::<Result<_,_>>()?,value.groups[1].iter().map(AlgebraicBlock::try_from).collect::<Result<_,_>>()?],value.a_factorization,value.schur_factorization,value.maximum_schur_entries,value.maximum_schur_products) }
}
impl PreconditionerPolicy {
    /// Capability family without erasing the policy's retained configuration.
    pub const fn kind(&self)->PreconditionerKind { match self { Self::Identity=>PreconditionerKind::Identity, Self::Jacobi=>PreconditionerKind::Jacobi, Self::UpperDiagonalSchur(_)=>PreconditionerKind::UpperDiagonalSchur } }
}

#[derive(Serialize,Deserialize)]
#[serde(rename_all="kebab-case",deny_unknown_fields)]
enum WireBlock { Field(String), ConstraintMultiplier { field:String } }
impl From<AlgebraicBlock> for WireBlock {
 fn from(value:AlgebraicBlock)->Self {match value { AlgebraicBlock::Field(field)=>Self::Field(field.ulid().to_string()), AlgebraicBlock::ConstraintMultiplier{field}=>Self::ConstraintMultiplier{field:field.ulid().to_string()} }}
}
impl TryFrom<&WireBlock> for AlgebraicBlock {
 type Error=Diagnostic;
 fn try_from(value:&WireBlock)->Result<Self,Diagnostic> {
 let text=match value {WireBlock::Field(field)|WireBlock::ConstraintMultiplier{field}=>field};
 let ulid=text.parse().map_err(|_|invalid_plan("block Field identity must be a canonical ULID"))?;
 let field=eqiora_core::Id::from_ulid(ulid);
 if field.ulid().to_string()!=*text {return Err(invalid_plan("block Field identity must be a canonical ULID"));}
 Ok(match value {WireBlock::Field(_)=>Self::Field(field),WireBlock::ConstraintMultiplier{..}=>Self::ConstraintMultiplier{field}})
 }
}
