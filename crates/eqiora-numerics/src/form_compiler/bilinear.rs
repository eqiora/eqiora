//! Mathematical contractions shared by scalar and mixed region integration.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Pairing {
    Value,
    ComponentValue { test: usize, trial: usize },
    Gradient,
    TestGradientTrialValue(usize),
    SkewGradient(usize),
    SymmetricGradient,
    Curl,
    Divergence,
    TestDivergenceTrialValue,
    TestValueTrialDivergence,
}

#[derive(Clone, Copy)]
pub(super) struct Basis<'a> {
    pub value: &'a [f64],
    pub gradient: &'a [f64],
}

impl Basis<'_> {
    fn divergence(self) -> f64 {
        let dimension = self.value.len();
        (0..dimension)
            .map(|axis| self.gradient[axis * dimension + axis])
            .sum()
    }
}

impl Pairing {
    pub(super) fn transpose(self) -> Option<Self> {
        Some(match self {
            Self::ComponentValue { test, trial } => Self::ComponentValue {
                test: trial,
                trial: test,
            },
            Self::TestDivergenceTrialValue => Self::TestValueTrialDivergence,
            Self::TestValueTrialDivergence => Self::TestDivergenceTrialValue,
            Self::TestGradientTrialValue(_) | Self::SkewGradient(_) => return None,
            other => other,
        })
    }

    pub(super) fn accepts(self, dimension: usize, test: usize, trial: usize) -> bool {
        match self {
            Self::Value | Self::Gradient => test == trial,
            Self::ComponentValue {
                test: row,
                trial: column,
            } => row < test && column < trial,
            Self::TestGradientTrialValue(axis) => axis < dimension && test == 1 && trial == 1,
            Self::SkewGradient(axis) => axis < dimension && test == trial,
            Self::Curl => dimension == 3 && test == 3 && trial == 3,
            Self::SymmetricGradient | Self::Divergence => test == dimension && trial == dimension,
            Self::TestDivergenceTrialValue => test == dimension && trial == 1,
            Self::TestValueTrialDivergence => test == 1 && trial == dimension,
        }
    }

    pub(super) fn entry(self, test: Basis<'_>, trial: Basis<'_>) -> f64 {
        match self {
            Self::Value => crate::affine_fem::dot(test.value, trial.value),
            Self::ComponentValue {
                test: row,
                trial: column,
            } => test.value[row] * trial.value[column],
            Self::Gradient => crate::affine_fem::dot(test.gradient, trial.gradient),
            Self::TestGradientTrialValue(axis) => test.gradient[axis] * trial.value[0],
            Self::SkewGradient(axis) => {
                let dimension = test.gradient.len() / test.value.len();
                (0..test.value.len())
                    .map(|component| {
                        test.gradient[component * dimension + axis] * trial.value[component]
                            - test.value[component] * trial.gradient[component * dimension + axis]
                    })
                    .sum()
            }
            Self::SymmetricGradient => {
                let dimension = test.value.len();
                let transpose = (0..dimension)
                    .flat_map(|i| {
                        (0..dimension).map(move |j| {
                            test.gradient[i * dimension + j] * trial.gradient[j * dimension + i]
                        })
                    })
                    .sum::<f64>();
                0.5 * (crate::affine_fem::dot(test.gradient, trial.gradient) + transpose)
            }
            Self::Curl => [(2, 1), (0, 2), (1, 0)]
                .into_iter()
                .map(|(i, j)| {
                    (test.gradient[3 * i + j] - test.gradient[3 * j + i])
                        * (trial.gradient[3 * i + j] - trial.gradient[3 * j + i])
                })
                .sum(),
            Self::Divergence => test.divergence() * trial.divergence(),
            Self::TestDivergenceTrialValue => test.divergence() * trial.value[0],
            Self::TestValueTrialDivergence => test.value[0] * trial.divergence(),
        }
    }
}
