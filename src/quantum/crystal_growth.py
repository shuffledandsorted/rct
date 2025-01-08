"""Crystal growth for computational patterns.

Physical crystals are nature's way of manifesting wave function patterns in 3D space.
Here we do the same thing, but manifesting computational wave functions as code.

Key concepts:
1. Energy minimization - patterns seek their most stable form
2. Symmetry - similar patterns reinforce each other
3. Phonons - vibrations in the pattern that can lead to new structures
4. Defects - imperfections that can actually make the crystal more useful
"""

import ast
import numpy as np
from typing import Optional, List, Dict, Set
from .ast_wave import ASTWaveFunction
from ..contracts.temporal import TemporalContract


class CrystalLattice:
    """A growing crystal lattice of computational patterns."""

    def __init__(self, dims: tuple[int, int] = (16, 32)):
        self.dims = dims
        self.wave_fn = ASTWaveFunction(dims)
        self.temperature = 1.0
        self.pressure = 0.7
        self.entropy = 0.0
        self.symmetries: Set[int] = set()  # Track pattern symmetries

    def add_seed(self, contract: TemporalContract) -> None:
        """Add a seed pattern to start crystal growth."""
        if not hasattr(contract, "wave_fn") or not contract.wave_fn:
            return

        # Initialize with seed pattern
        self.wave_fn.encode_node(self._extract_pattern(contract))
        # Record initial symmetries
        self.symmetries.add(hash(str(contract.psi)))

    def _extract_pattern(self, contract: TemporalContract) -> ast.AST:
        """Extract the computational pattern from a contract."""
        if callable(contract.psi):
            # Get the actual transformation pattern
            pattern = contract.psi(np.zeros(self.dims))
            # Convert to AST structure
            return self._pattern_to_ast(pattern)
        return ast.Pass()  # Default empty pattern

    def _pattern_to_ast(self, pattern: np.ndarray) -> ast.AST:
        """Convert a numpy pattern to AST structure."""
        # Find dominant frequencies in the pattern
        freqs = np.fft.fft2(pattern)
        main_freq = np.unravel_index(np.argmax(np.abs(freqs)), freqs.shape)

        # Convert frequency components to computational structure
        if np.abs(freqs[main_freq]) > 0.8:  # Strong periodic pattern
            return ast.For(
                target=ast.Name(id="i", ctx=ast.Store()),
                iter=ast.Call(
                    func=ast.Name(id="range", ctx=ast.Load()),
                    args=[ast.Constant(value=int(main_freq[0]))],
                    keywords=[],
                ),
                body=[
                    ast.Expr(
                        ast.Call(
                            func=ast.Name(id="transform", ctx=ast.Load()),
                            args=[ast.Name(id="state", ctx=ast.Load())],
                            keywords=[],
                        )
                    )
                ],
                orelse=[],
            )
        else:  # More like a point transformation
            return ast.Expr(
                ast.Call(
                    func=ast.Name(id="transform", ctx=ast.Load()),
                    args=[ast.Name(id="state", ctx=ast.Load())],
                    keywords=[],
                )
            )

    def apply_pressure(self, amount: float = 0.1) -> None:
        """Apply pressure to encourage pattern alignment."""
        self.pressure += amount
        # Increase phase coherence requirements
        self.wave_fn.amplitude *= np.exp(
            1j * self.pressure * np.angle(self.wave_fn.amplitude)
        )

    def add_thermal_energy(self, amount: float = 0.1) -> None:
        """Add thermal energy to explore new configurations."""
        self.temperature += amount
        # Add random phase fluctuations
        noise = np.random.normal(0, self.temperature, self.dims)
        self.wave_fn.amplitude *= np.exp(1j * noise)

    def measure_entropy(self) -> float:
        """Measure the computational entropy of the crystal."""
        # Entropy increases with:
        # 1. Temperature (thermal entropy)
        # 2. Number of symmetries (configurational entropy)
        # 3. Pattern complexity (information entropy)
        amplitude = np.abs(self.wave_fn.amplitude).astype(np.float64)
        thermal = self.temperature * np.log(amplitude.sum() + 1e-10)
        config = np.log(len(self.symmetries) + 1)
        info = np.sum(amplitude * np.log(amplitude + 1e-10), dtype=np.float64)
        self.entropy = float(thermal + config - info)  # Note: info entropy is negative
        return self.entropy

    def grow(self, steps: int = 100) -> Optional[ast.AST]:
        """Grow the crystal through simulated annealing."""
        best_pattern = None
        best_coherence = 0.0

        for step in range(steps):
            # Current state
            coherence = self.wave_fn.measure_coherence()
            if coherence > best_coherence:
                best_coherence = coherence
                best_pattern = self.wave_fn.decode_ast()

            # Adjust conditions based on entropy
            entropy = self.measure_entropy()
            if entropy > 2.0:  # Too disordered
                self.temperature *= 0.95  # Cool down
                self.apply_pressure(0.05)  # Increase pressure
            else:  # Too ordered
                self.add_thermal_energy(0.05)  # Heat up

            # Let the system evolve
            self.wave_fn.amplitude *= np.exp(-step / steps)  # Gradual decay

            # Only stop if we've found an exceptionally good pattern
            # and we're at least halfway through the steps
            if best_coherence > max(self.pressure * 1.5, 0.9) and step > steps // 2:
                break

        return best_pattern

    def get_symmetries(self) -> List[str]:
        """Get the symmetries found in the crystal."""
        return [f"Symmetry {i}" for i, _ in enumerate(self.symmetries)]


def grow_computational_crystal(
    seed_contract: TemporalContract,
    temperature: float = 1.0,
    pressure: float = 0.7,
    steps: int = 100,
) -> Optional[ast.AST]:
    """Grow a computational crystal from a seed contract.

    This is the main interface for crystal growth, hiding the complexity
    of the physical simulation.
    """
    lattice = CrystalLattice()
    lattice.temperature = temperature
    lattice.pressure = pressure

    # Plant the seed
    lattice.add_seed(seed_contract)

    # Grow the crystal
    return lattice.grow(steps)
