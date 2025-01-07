import numpy as np
from typing import Tuple, Optional, List
from scipy.linalg import sqrtm

class DensityMatrix:
    """Represents a quantum state as a density matrix in probability space"""
    
    def __init__(self, dims: Tuple[int, int], pure_state: Optional[np.ndarray] = None):
        """Initialize density matrix from pure state or as maximally mixed state"""
        self.dims = dims
        if pure_state is not None:
            # Convert pure state to density matrix
            pure_state = pure_state.reshape(-1)
            self.rho = np.outer(pure_state, np.conj(pure_state))
        else:
            # Start with maximally mixed state
            n = np.prod(dims)
            self.rho = np.eye(n) / n
            
    def get_probability_distribution(self) -> np.ndarray:
        """Get probability distribution in computational basis"""
        return np.real(np.diag(self.rho)).reshape(self.dims)
        
    def von_neumann_entropy(self) -> float:
        """Calculate von Neumann entropy S = -Tr(ρ ln ρ)"""
        eigenvalues = np.linalg.eigvalsh(self.rho)
        # Remove very small eigenvalues to avoid log(0)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        return float(-np.sum(eigenvalues * np.log(eigenvalues)))
        
    def purity(self) -> float:
        """Calculate purity Tr(ρ²)"""
        return float(np.trace(self.rho @ self.rho).real)
        
    def fidelity(self, other: 'DensityMatrix') -> float:
        """Calculate fidelity between two mixed states"""
        # F(ρ,σ) = Tr(√(√ρ σ √ρ))
        sqrt_rho = sqrtm(self.rho)
        return float(np.trace(sqrtm(sqrt_rho @ other.rho @ sqrt_rho)).real)

    def partial_trace(self, subsystem: int) -> 'DensityMatrix':
        """Compute reduced density matrix by tracing out subsystem"""
        n = int(np.sqrt(self.rho.shape[0]))
        if subsystem == 0:
            # Trace out first subsystem
            reduced = np.zeros((n, n), dtype=complex)
            for i in range(n):
                reduced += self.rho[i::n, i::n]
        else:
            # Trace out second subsystem
            reduced = np.zeros((n, n), dtype=complex)
            for i in range(n):
                reduced += self.rho[i*n:(i+1)*n, i*n:(i+1)*n]
        
        return DensityMatrix(dims=(n,n), pure_state=reduced)

def quantum_relative_entropy(rho: DensityMatrix, sigma: DensityMatrix) -> float:
    """Calculate quantum relative entropy S(ρ||σ) = Tr(ρ(ln ρ - ln σ))"""
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
    """Calculate p-Wasserstein distance between quantum states"""
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
    """Manages probability distributions and transformations in quantum state space"""
    
    def __init__(self, dims: Tuple[int, int]):
        self.dims = dims
        
    def pure_to_mixed(self, pure_state: np.ndarray) -> DensityMatrix:
        """Convert pure state to density matrix representation"""
        return DensityMatrix(self.dims, pure_state)
        
    def measure_observables(self, state: DensityMatrix, 
                          observables: List[np.ndarray]) -> List[float]:
        """Measure quantum observables in probability space"""
        return [float(np.trace(state.rho @ obs).real) for obs in observables]
        
    def quantum_channel(self, state: DensityMatrix, 
                       kraus_operators: List[np.ndarray]) -> DensityMatrix:
        """Apply quantum channel through Kraus operators"""
        result = np.zeros_like(state.rho)
        for K in kraus_operators:
            result += K @ state.rho @ K.conj().T
        new_state = DensityMatrix(self.dims)
        new_state.rho = result
        return new_state 