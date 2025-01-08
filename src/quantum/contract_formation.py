"""Quantum contract formation through state gathering and superposition.

This module implements the fundamental operation of quantum RCT:
gathering valid states from different sources and forming contracts through
superposition. The process maintains energy conservation while allowing
quantum dynamics to find equilibrium points representing stable agreements.

The operation follows these key steps:
1. Gather quantum states from registered providers
2. Align and normalize states while preserving structure
3. Form superposition through cooperative collapse
4. Let quantum dynamics find equilibrium

This creates a unified framework for combining different perspectives
while maintaining quantum mechanical constraints.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .utils import quantum_normalize, calculate_cohesion, calculate_geodesic_collapse
from .game_theory import cooperative_collapse


@dataclass
class ValidState:
    """A quantum state that can participate in contract formation.

    Represents a valid quantum state from a particular source that can
    contribute to contract formation. Includes metadata about the state's
    origin and relative importance.

    Attributes:
        state: The quantum state as a complex numpy array
        source: String identifier of where this state came from
        base_weight: Base importance weight for this state
        metadata: Additional information about the state
    """

    state: np.ndarray
    source: str
    base_weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class QuantumContractFormation:
    """Implements the fundamental quantum RCT operation.

    This class handles the core operation of quantum RCT:
    gathering valid states from different sources and forming contracts
    through superposition. The process maintains energy conservation
    while allowing quantum dynamics to find equilibrium points.

    The operation has four key steps:
    1. Gather quantum states from registered providers
    2. Align and normalize states while preserving structure
    3. Form superposition through cooperative collapse
    4. Let quantum dynamics find equilibrium

    This creates stable contracts that represent agreements between
    different quantum states while maintaining physical constraints.
    """

    def __init__(self, dims: Tuple[int, int]):
        """Initialize contract formation for given dimensions.

        Args:
            dims: Tuple of (height, width) for the quantum states
        """
        self.dims = dims
        self._state_providers = {}

    def register_state_provider(self, name: str, provider_fn, base_weight: float = 1.0):
        """Register a function that can provide quantum states.

        State providers are called during contract formation to gather
        relevant quantum states based on the current context.

        Args:
            name: Unique identifier for this provider
            provider_fn: Function that takes context and returns states
            base_weight: Base importance weight for states from this provider
        """
        self._state_providers[name] = (provider_fn, base_weight)

    def gather_valid_states(self, context: Dict[str, Any]) -> List[ValidState]:
        """Gather all valid quantum states from registered providers.

        This is the first step of contract formation. Each provider
        examines the context and returns relevant states that should
        participate in the contract.

        The gathering process:
        1. Calls each registered provider with the context
        2. Validates and normalizes returned states
        3. Packages states with metadata about their source

        Args:
            context: Dictionary of contextual information

        Returns:
            List of ValidState objects ready for contract formation
        """
        valid_states = []

        for name, (provider_fn, base_weight) in self._state_providers.items():
            states = provider_fn(context)
            if not isinstance(states, list):
                states = [states] if states is not None else []

            for state in states:
                if isinstance(state, np.ndarray):
                    valid_states.append(
                        ValidState(state=state, source=name, base_weight=base_weight)
                    )

        return valid_states

    def align_and_normalize(
        self, states: List[ValidState]
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Align dimensions and normalize states while calculating weights.

        Prepares states for superposition by ensuring they're in the same
        Hilbert space and properly normalized. The process:
        1. Aligns dimensions through geodesic collapse
        2. Normalizes states while preserving phase relationships
        3. Calculates weights based on coherence and base importance

        Args:
            states: List of ValidState objects to prepare

        Returns:
            Tuple of (normalized_states, weights) ready for superposition
        """
        normalized_states = []
        weights = []

        for valid_state in states:
            # Align dimensions
            if valid_state.state.size != np.prod(self.dims):
                aligned = calculate_geodesic_collapse(
                    valid_state.state.reshape(-1),
                    np.zeros(np.prod(self.dims), dtype=np.complex128),
                ).reshape(self.dims)
            else:
                aligned = valid_state.state.reshape(self.dims)

            # Normalize
            normalized = quantum_normalize(aligned)
            normalized_states.append(normalized)

            # Calculate weight based on coherence and base weight
            coherence = calculate_cohesion([normalized])
            weight = (
                valid_state.base_weight * np.exp(coherence) / (1 + np.exp(coherence))
            )
            weights.append(weight)

        # Normalize weights
        if weights:
            total = sum(weights)
            weights = [w / total for w in weights]

        return normalized_states, weights

    def form_contract(self, context: Dict[str, Any]) -> Optional[np.ndarray]:
        """Form a contract through the fundamental quantum RCT operation.

        This is the core method implementing quantum contract formation:
        1. Gather valid states from all providers
        2. Align and normalize states while preserving structure
        3. Form superposition through cooperative collapse
        4. Let quantum dynamics find equilibrium

        The operation maintains energy conservation across all states,
        including any previous state provided in the context. The resulting
        contract represents a stable agreement between the quantum states.

        Args:
            context: Dictionary containing contextual information and
                    optional previous state

        Returns:
            Combined quantum state representing the formed contract,
            or None if no valid states are available
        """
        # Gather states
        valid_states = self.gather_valid_states(context)
        if not valid_states:
            return None

        # Add previous state if it exists
        previous_state = context.get("previous_state")
        if previous_state is not None:
            valid_states.append(
                ValidState(
                    state=previous_state,
                    source="previous_state",
                    base_weight=0.5,  # Equal weight to new information
                )
            )

        # Align and normalize while preserving total energy
        normalized_states, weights = self.align_and_normalize(valid_states)

        # Calculate initial total energy
        initial_energy = sum(np.sum(np.abs(state) ** 2) for state in normalized_states)

        # Form superposition through cooperative collapse
        combined_state = cooperative_collapse(normalized_states, weights=weights)

        # Rescale to preserve initial energy
        final_energy = np.sum(np.abs(combined_state) ** 2)
        energy_scale = np.sqrt(initial_energy / final_energy)
        combined_state *= energy_scale

        return combined_state
