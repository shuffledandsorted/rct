import numpy as np
from typing import List
from scipy.signal import convolve2d


class QuantumOperator:
    """
    Represents a measurement operator with quantum-like properties:
    - Hermitian (self-adjoint)
    - Normalized (preserves probability)
    - Linear
    """

    def __init__(self, matrix: np.ndarray):
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Operator must be a numpy array")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Operator must be square")

        # Convert to complex if needed
        self.matrix = matrix.astype(np.complex128)
        self._make_hermitian()
        self._normalize()

    def _make_hermitian(self):
        """Ensure operator is Hermitian (self-adjoint)"""
        self.matrix = 0.5 * (self.matrix + np.conj(self.matrix.T))

    def _normalize(self):
        """Normalize to preserve probability"""
        self.matrix /= np.sqrt(np.sum(np.abs(self.matrix) ** 2))

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Apply the operator to a quantum state"""
        return convolve2d(state, self.matrix, mode="same")

    @classmethod
    def sobel_vertical(cls) -> "QuantumOperator":
        """Factory method for vertical edge detection"""
        return cls(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))

    @classmethod
    def sobel_horizontal(cls) -> "QuantumOperator":
        """Factory method for horizontal edge detection"""
        return cls(np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))

    @classmethod
    def diagonal(cls) -> "QuantumOperator":
        """Factory method for diagonal pattern detection"""
        return cls(np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]))

    @property
    def is_hermitian(self) -> bool:
        """Check if operator is still Hermitian"""
        return np.allclose(self.matrix, np.conj(self.matrix.T))

    @classmethod
    def from_patterns(cls, patterns: List[np.ndarray]) -> "QuantumOperator":
        """Create operator from learned patterns"""
        if not patterns:
            raise ValueError("Need at least one pattern")

        reference = patterns[0]
        aligned = []

        for pattern in patterns:
            phase_diff = np.angle(pattern * np.conj(reference))
            aligned.append(pattern * np.exp(-1j * phase_diff))

        avg_pattern = np.mean(aligned, axis=0)
        return cls(avg_pattern) 