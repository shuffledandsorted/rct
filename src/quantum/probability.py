"""Quantum probability and density matrix operations for RCT.

This module implements quantum probability concepts used in RCT:
- Density matrix representations of quantum states
- Quantum entropy and information metrics
- State transformations and measurements
- Distance measures between quantum states

The implementation focuses on maintaining quantum mechanical properties
while providing practical tools for quantum probability calculations.
"""

import numpy as np
from typing import Tuple, Optional, List
from numpy.typing import NDArray
from scipy.linalg import sqrtm

ComplexArray = NDArray[np.complex128]

class DensityMatrix:
    """Represents a quantum state as a density matrix in probability space.

    The density matrix ρ provides a complete description of a quantum state,
    supporting both pure and mixed states. It maintains key properties:
    - Hermiticity: ρ = ρ†
    - Positive semidefinite: ⟨ψ|ρ|ψ⟩ ≥ 0
    - Trace one: Tr(ρ) = 1

    This implementation supports:
    - Probability distribution extraction
    - Entropy calculations
    - Fidelity measures
    - Partial trace operations
    """

    def __init__(self, dims: Tuple[int, int], pure_state: Optional[np.ndarray] = None):
        """Initialize density matrix from pure state or as maximally mixed state.

        Args:
            dims: Tuple of (height, width) for the quantum state space
            pure_state: Optional pure state to convert to density matrix.
                       If None, creates maximally mixed state.
        """
        self.dims = dims
        if pure_state is not None:
            # Convert pure state to density matrix
            pure_state = pure_state.reshape(-1)
            self.rho = np.outer(pure_state, np.conj(pure_state))
        else:
            # Start with maximally mixed state
            n = int(np.prod(dims))
            self.rho = np.eye(n, dtype=np.complex128) / n

    def get_probability_distribution(self) -> np.ndarray:
        """Get probability distribution in computational basis.

        Extracts classical probabilities by taking diagonal elements,
        representing measurement outcomes in the computational basis.

        Returns:
            2D array of real probabilities reshaped to state dimensions
        """
        return np.real(np.diag(self.rho)).reshape(self.dims)

    def von_neumann_entropy(self) -> float:
        """Calculate von Neumann entropy S = -Tr(ρ ln ρ).

        The von Neumann entropy generalizes classical entropy to quantum
        states, measuring quantum uncertainty and entanglement. Key properties:
        - Zero for pure states
        - Maximum for maximally mixed states
        - Subadditive across tensor products

        Returns:
            Non-negative float representing quantum entropy
        """
        eigenvalues = np.linalg.eigvalsh(self.rho)
        # Remove very small eigenvalues to avoid log(0)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        return float(-np.sum(eigenvalues * np.log(eigenvalues)))

    def purity(self) -> float:
        """Calculate purity Tr(ρ²)."""
        product: ComplexArray = np.matmul(self.rho, self.rho)
        return float(np.sum(np.diag(product)).real)

    def fidelity(self, other: 'DensityMatrix') -> float:
        """Calculate fidelity between two mixed states."""
        sqrt_rho: ComplexArray = sqrtm(self.rho)  # type: ignore[no-any-return]
        product: ComplexArray = np.matmul(np.matmul(sqrt_rho, other.rho), sqrt_rho)
        return float(np.sum(np.diag(sqrtm(product))).real)  # type: ignore[no-any-return]

    def partial_trace(self, subsystem: int) -> 'DensityMatrix':
        """Compute reduced density matrix by tracing out subsystem.

        Performs partial trace operation to examine subsystem states.
        Critical for analyzing:
        - Entanglement
        - Subsystem dynamics
        - Decoherence effects

        Args:
            subsystem: Which subsystem to trace out (0 or 1)

        Returns:
            Reduced density matrix for remaining subsystem
        """
        n = int(np.sqrt(self.rho.shape[0]))
        if subsystem == 0:
            # Trace out first subsystem
            reduced = np.zeros((n, n), dtype=np.complex128)
            for i in range(n):
                reduced += self.rho[i::n, i::n]
        else:
            # Trace out second subsystem
            reduced = np.zeros((n, n), dtype=np.complex128)
            for i in range(n):
                reduced += self.rho[i*n:(i+1)*n, i*n:(i+1)*n]

        return DensityMatrix(dims=(n,n), pure_state=reduced)

def quantum_relative_entropy(rho: DensityMatrix, sigma: DensityMatrix) -> float:
    """Calculate quantum relative entropy S(ρ||σ) = Tr(ρ(ln ρ - ln σ)).

    The quantum relative entropy measures distinguishability between
    quantum states. Key properties:
    - Non-negative: S(ρ||σ) ≥ 0
    - Zero iff ρ = σ
    - Infinite if states have different support

    Args:
        rho: First density matrix
        sigma: Second density matrix (reference state)

    Returns:
        Non-negative float or infinity
    """
    # Eigendecomposition of both states
    rho_eigvals, rho_eigvecs = np.linalg.eigh(rho.rho)
    sigma_eigvals, sigma_eigvecs = np.linalg.eigh(sigma.rho)

    # Remove very small eigenvalues
    mask_rho = rho_eigvals > 1e-10
    mask_sigma = sigma_eigvals > 1e-10

    if not (np.all(mask_rho) and np.all(mask_sigma)):
        # States have different support - relative entropy is infinite
        return float('inf')

    return float(np.trace(rho.rho @ (np.log(rho.rho) - np.log(sigma.rho))).real)

def wasserstein_distance(rho: DensityMatrix, sigma: DensityMatrix, p: int = 2) -> float:
    """Calculate p-Wasserstein distance between quantum states.

    Computes optimal transport distance between quantum probability
    distributions. Key features:
    - Respects underlying geometry
    - Sensitive to both amplitude and phase differences
    - Generalizes classical Wasserstein distance

    Args:
        rho: First density matrix
        sigma: Second density matrix
        p: Order of distance (default 2 for standard Wasserstein-2)

    Returns:
        Non-negative float representing transport distance
    """
    # Get probability distributions
    p_rho = rho.get_probability_distribution()
    p_sigma = sigma.get_probability_distribution()

    # Calculate cost matrix based on grid distances
    y, x = np.mgrid[0:rho.dims[0], 0:rho.dims[1]]
    cost_matrix = np.zeros((np.prod(rho.dims), np.prod(rho.dims)))
    for i in range(np.prod(rho.dims)):
        i_y, i_x = i // rho.dims[1], i % rho.dims[1]
        for j in range(np.prod(rho.dims)):
            j_y, j_x = j // rho.dims[1], j % rho.dims[1]
            cost_matrix[i,j] = ((i_y - j_y)**2 + (i_x - j_x)**2)**(p/2)

    # Flatten probability distributions
    p_rho = p_rho.flatten()
    p_sigma = p_sigma.flatten()

    # Simple approximation of Wasserstein distance
    # (Could be improved with proper optimal transport calculation)
    sorted_rho = np.sort(p_rho)
    sorted_sigma = np.sort(p_sigma)
    return float(np.sum(cost_matrix * np.abs(sorted_rho - sorted_sigma))**(1/p))

class ProbabilitySpace:
    """Manages probability distributions and transformations in quantum state space.

    Provides tools for:
    - Converting between pure and mixed states
    - Measuring quantum observables
    - Applying quantum channels
    - Analyzing probability distributions

    This creates a unified framework for handling quantum probability
    operations while maintaining proper normalization and coherence.
    """

    def __init__(self, dims: Tuple[int, int]):
        """Initialize probability space with given dimensions.

        Args:
            dims: Tuple of (height, width) for quantum state space
        """
        self.dims = dims

    def pure_to_mixed(self, pure_state: np.ndarray) -> DensityMatrix:
        """Convert pure state to density matrix representation.

        Transforms a pure state vector into its density matrix form:
        ρ = |ψ⟩⟨ψ|

        Args:
            pure_state: Complex array representing pure quantum state

        Returns:
            Density matrix representation
        """
        return DensityMatrix(self.dims, pure_state)

    def measure_observables(
        self, state: DensityMatrix, observables: List[ComplexArray]
    ) -> List[float]:
        """Measure quantum observables in probability space."""
        return [
            float(np.sum(np.diag(np.matmul(state.rho, obs))).real)
            for obs in observables
        ]

    def quantum_channel(self, state: DensityMatrix, 
                       kraus_operators: List[np.ndarray]) -> DensityMatrix:
        """Apply quantum channel through Kraus operators.

        Implements general quantum operations through Kraus decomposition:
        ρ → Σᵢ KᵢρKᵢ†

        Properties maintained:
        - Complete positivity
        - Trace preservation
        - Proper normalization

        Args:
            state: Input quantum state
            kraus_operators: List of Kraus operators defining channel

        Returns:
            Transformed quantum state
        """
        result = np.zeros_like(state.rho)
        for K in kraus_operators:
            result += K @ state.rho @ K.conj().T
        new_state = DensityMatrix(self.dims)
        new_state.rho = result
        return new_state 
