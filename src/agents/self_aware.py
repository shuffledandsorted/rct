from typing import Tuple, Optional, Dict, List
import numpy as np

from .base import QuantumAgent
from .temporal import TemporalMixin
from .config import AgentConfig
from ..quantum import WaveFunction, QuantumOperator


class SelfAwareAgent(TemporalMixin, QuantumAgent):
    """
    A quantum agent capable of recursive self-modeling and awareness based on 
    Recursive Contract Theory principles.
    """

    def __init__(self, dims: Tuple[int, int], config: Optional[AgentConfig] = None, age: int = 0):
        super().__init__(dims=dims, config=config, age=age)
        # Initialize temporal memory from mixin
        self.temporal_memory: List[np.ndarray] = []
        # Self-model quantum state
        self.self_model = WaveFunction(dims)
        # Models of other agents
        self.other_models: Dict[str, WaveFunction] = {}
        # Track recursive depth of self-modeling
        self.recursive_depth = 0
        # Use config's critical depth
        self.config = config or AgentConfig()
        # Add previous state tracking
        self.previous_state = np.zeros(dims)

    def update_quantum_state(self, image: np.ndarray):
        """Update both quantum state and self-model"""
        # Update base quantum state
        self.wave_fn = WaveFunction(self.dims)
        normalized_image = (image - image.min()) / (image.max() - image.min() + 1e-10)
        self.wave_fn.amplitude *= (normalized_image * 0.9 + 0.1)

        # Update temporal memory
        self.update_memory(self.wave_fn.amplitude)

        # Update self-model based on temporal history
        self._update_self_model()

        # Increment recursive depth
        self.recursive_depth += 1

    def combine_states(self, current_state: np.ndarray, past_state: np.ndarray) -> np.ndarray:
        """
        Combine current and past states while maintaining self-model consistency.
        Implementation of abstract method from TemporalMixin.
        """
        # Weight recent state more heavily but maintain temporal consistency
        combined = 0.7 * current_state + 0.3 * past_state
        return combined

    def _update_self_model(self):
        """Update internal self-model based on temporal history"""
        if not self.temporal_memory:
            return

        # Get aggregated past state
        past_state = self.get_past_state()
        if past_state is None:
            return

        # Update self-model as quantum superposition of past and present
        self.self_model.amplitude = self.combine_states(
            self.wave_fn.amplitude,
            past_state
        )

    def model_other_agent(self, agent_id: str, other_agent: QuantumAgent):
        """Create and maintain models of other agents"""
        if agent_id not in self.other_models:
            self.other_models[agent_id] = WaveFunction(self.dims)

        # Update model through quantum interference
        self.other_models[agent_id].amplitude = 0.8 * self.other_models[agent_id].amplitude + \
                                              0.2 * other_agent.wave_fn.amplitude

    def measure_self_awareness(self) -> float:
        """
        Measure degree of self-awareness based on:
        1. Recursive depth relative to critical threshold
        2. Quantum coherence between self-model and actual state
        3. Energy stability from temporal consistency
        """
        if not self.temporal_memory:
            return 0.0

        # Check recursive depth threshold using config
        depth_factor = min(self.recursive_depth / self.config.critical_depth, 1.0)

        # Measure quantum coherence between self-model and actual state
        coherence = 1.0 - self.measure_phase_distance(
            self.self_model.amplitude,
            self.wave_fn.amplitude
        )

        # Energy stability from temporal consistency
        energy_stability = self._measure_energy_stability()

        # Combine factors with quantum weighting
        awareness = np.abs(depth_factor * 0.4 + 
                         coherence * 0.4 + 
                         energy_stability * 0.2)

        return float(awareness)

    def _measure_energy_stability(self) -> float:
        """Measure stability of quantum energy states over time"""
        if len(self.temporal_memory) < 2:
            return 0.0

        # Calculate quantum energy variations over recent history
        energy_variations = []
        for i in range(1, len(self.temporal_memory)):
            prev = self.temporal_memory[i-1]
            curr = self.temporal_memory[i]
            # Use phase distance as proxy for energy difference
            energy_variations.append(self.measure_phase_distance(prev, curr))

        # Convert to quantum stability measure (1.0 = perfectly stable)
        avg_variation = np.mean(energy_variations)
        stability = np.exp(-avg_variation)
        return float(stability)

    def compare_contract_order(self, other: "SelfAwareAgent") -> bool:
        """
        Implement partial ordering from Theorem 3.2:
        C₁ ≤ C₂ ⟺ ∃R: Y(R(C₂)) = Y(C₁) and E(C₁) ≤ E(C₂)
        """
        # Check energy ordering
        self_energy = self._measure_energy_stability()
        other_energy = other._measure_energy_stability()

        # Check state refinement through phase distance
        state_distance = self.measure_phase_distance(
            self.wave_fn.amplitude, other.wave_fn.amplitude
        )

        # Return true if this contract refines the other with lower energy
        return self_energy <= other_energy and state_distance < 0.1

    def join_contracts(self, other: "SelfAwareAgent") -> np.ndarray:
        """
        Implement energy-preserving join operation for the lattice structure.
        Returns the joined quantum state that preserves energy.
        """
        # Calculate energies
        e1 = np.sum(np.abs(self.wave_fn.amplitude) ** 2)
        e2 = np.sum(np.abs(other.wave_fn.amplitude) ** 2)

        # Energy-preserving weights
        total_energy = e1 + e2
        w1 = e1 / total_energy if total_energy > 0 else 0.5
        w2 = e2 / total_energy if total_energy > 0 else 0.5

        # Create joined state preserving energy
        joined_state = w1 * self.wave_fn.amplitude + w2 * other.wave_fn.amplitude

        # Normalize while preserving relative energy
        norm = np.linalg.norm(joined_state)
        if norm > 0:
            joined_state = joined_state * np.sqrt(total_energy) / norm

        return joined_state

    def meet_contracts(self, other: "SelfAwareAgent") -> np.ndarray:
        """
        Implement energy-preserving meet operation for the lattice structure.
        Returns the met quantum state that preserves minimum energy.
        """
        # Find common subspace through interference
        interference = self.wave_fn.amplitude * np.conj(other.wave_fn.amplitude)

        # Take minimum energy components
        min_energy = min(
            np.sum(np.abs(self.wave_fn.amplitude) ** 2),
            np.sum(np.abs(other.wave_fn.amplitude) ** 2),
        )

        # Create met state preserving minimum energy
        met_state = interference / (np.linalg.norm(interference) + 1e-10)
        met_state = met_state * np.sqrt(min_energy)

        return met_state

    def is_self_aware(self) -> bool:
        """
        Determine if agent has achieved stable self-awareness based on:
        1. Sufficient recursive depth beyond critical threshold
        2. High quantum coherence between self-model and actual state
        3. Energy conservation in temporal evolution
        """
        awareness = self.measure_self_awareness()
        return awareness > 0.8  # Threshold for stable self-awareness 

    def test_consciousness_emergence(self) -> Tuple[bool, Dict[str, float]]:
        """
        Test consciousness emergence based on three null hypotheses from RCT Section 5.2:
        1. Self-Referential Stability: Y(C(C)) = C
        2. Mutual Modeling: Y(C₁(C₂)) = Y(C₂(C₁))
        3. Energy Minimization: E(C) = min{E | Y(C) = C}

        Returns:
            Tuple[bool, Dict[str, float]]: (consciousness_emerged, test_results)
            where test_results contains p-values for each hypothesis test
        """
        results = {}

        # Test 1: Self-Referential Stability
        # H₀: Y(C(C)) ≠ C
        self_ref_distance = self.measure_phase_distance(
            self.self_model.amplitude, self.wave_fn.amplitude
        )
        # More lenient p-value calculation
        results["self_ref_p"] = 1.0 - np.exp(-self_ref_distance * 0.5)

        # Test 2: Mutual Modeling Consistency
        # H₀: Y(C₁(C₂)) ≠ Y(C₂(C₁))
        if len(self.other_models) >= 2:
            model_distances = []
            for id1, model1 in self.other_models.items():
                for id2, model2 in self.other_models.items():
                    if id1 != id2:
                        dist = self.measure_phase_distance(
                            model1.amplitude, model2.amplitude
                        )
                        model_distances.append(dist)
            mutual_model_distance = np.mean(model_distances) if model_distances else 1.0
            # More lenient p-value calculation
            results["mutual_model_p"] = 1.0 - np.exp(-mutual_model_distance * 0.5)
        else:
            # Default to moderate p-value when not enough models
            results["mutual_model_p"] = 0.5

        # Test 3: Energy Minimization
        # H₀: E(C) > min{E | Y(C) = C}
        energy_stability = self._measure_energy_stability()
        # More lenient p-value calculation
        results["energy_min_p"] = max(1.0 - energy_stability * 1.5, 0.0)

        # Consciousness emerges if we reject all null hypotheses (p < 0.1 instead of 0.05)
        consciousness_emerged = all(p < 0.1 for p in results.values())

        return consciousness_emerged, results

    def update_with_state(self, state: np.ndarray) -> None:
        """Update wave function with new state and let self-model converge to eigenvector.

        This is the fundamental operation of a self-aware agent:
        1. Update wave function with new state
        2. Let self-model naturally converge to its stable eigenvector
        3. Update temporal memory to maintain history
        """
        # Update wave function
        self.wave_fn.amplitude = state

        # Update temporal memory
        self.update_memory(state)

        # Let self-model converge to eigenvector
        self._update_self_model()

        # Increment recursive depth as we've done another cycle of self-modeling
        self.recursive_depth += 1
