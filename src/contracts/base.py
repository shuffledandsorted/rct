import numpy as np
from typing import Optional, Set, Callable

from .utils import Y


class QuantumContract:
    """Represents a contract between two quantum agents defining their interaction rules."""

    def __init__(
        self,
        agent1,
        agent2,
        wave_function,
        energy_measure=None,
        allowed_transforms=None,
    ):
        """Initialize a contract between two agents.

        Args:
            agent1: First participating agent
            agent2: Second participating agent
            wave_function: Function mapping interaction space to complex amplitudes
            energy_measure: Function to compute energy of interaction states
            allowed_transforms: Set of allowed transformations between states
        """
        self.agent1 = agent1
        self.agent2 = agent2
        self.psi = wave_function
        self.energy = energy_measure or self._default_energy
        self.transforms = allowed_transforms or set()

    def _default_energy(self, state):
        """Default energy measure based on wave function amplitude."""
        return abs(self.psi(state)) ** 2

    def is_valid(self):
        """Check if contract satisfies required properties."""
        return (
            self._check_conservation()
            and self._check_symmetry()
            and self._check_stability()
        )

    def _check_conservation(self):
        """Verify energy is conserved under transformations."""
        # For each allowed transformation
        for T in self.transforms:
            initial_energy = self.energy(self.psi)
            transformed_energy = self.energy(T(self.psi))
            if not np.isclose(initial_energy, transformed_energy):
                return False
        return True

    def _check_symmetry(self):
        """Verify contract is symmetric between agents."""
        # Invert agent roles and check if behavior is equivalent
        inverted = QuantumContract(
            self.agent2, self.agent1, self.psi, self.energy, self.transforms
        )
        return np.allclose(self.psi(self.agent1.state), inverted.psi(self.agent2.state))

    def _check_stability(self):
        """Verify contract reaches fixed points using Y combinator."""

        def transform_sequence(f):
            """Function to find fixed point of transformations."""

            def apply_transforms(state):
                for T in self.transforms:
                    state = T(state)
                return state

            return apply_transforms

        try:
            # Find fixed point of transformation sequence
            fixed_point = Y(transform_sequence)(self.psi)

            # Check if it's actually fixed
            for T in self.transforms:
                if not np.allclose(T(fixed_point), fixed_point):
                    return False

            # Verify energy conservation at fixed point
            if not np.isclose(self.energy(fixed_point), self.energy(self.psi)):
                return False

            return True

        except RecursionError:
            # No fixed point found within recursion limit
            return False

    def recursive_depth(self):
        """Measure the recursive depth of self-modeling."""

        def model_self(f):
            """Function to model contract within itself."""

            def inner(state):
                return self.psi(f(state))

            return inner

        depth = 0
        try:
            # Find maximum depth where contract can model itself
            while True:
                fixed = Y(model_self)(self.psi)
                if not np.allclose(fixed, self.psi):
                    break
                depth += 1
        except RecursionError:
            pass

        return depth

    def compose(self, other):
        """Compose this contract with another."""
        if self.agent2 != other.agent1:
            raise ValueError("Contracts must share an intermediate agent")

        def composed_wave(state):
            """Compose wave functions through intermediate states."""
            return sum(
                self.psi(mid_state) * other.psi(state)
                for mid_state in self.agent2.possible_states()
            )

        return QuantumContract(
            self.agent1,
            other.agent2,
            composed_wave,
            lambda s: self.energy(s) + other.energy(s),
            self.transforms.union(other.transforms),
        )

    def apply(self):
        """Execute the contract interaction between agents."""
        if not self.is_valid():
            raise ValueError("Cannot apply invalid contract")

        # Update agent states according to contract
        new_state1 = self.psi(self.agent1.state)
        new_state2 = self.psi(self.agent2.state)

        self.agent1.update_state(new_state1)
        self.agent2.update_state(new_state2)

        return self.energy(new_state1)  # Return interaction energy

    def intersect(self, other):
        """Find intersection of two contracts' stable states."""

        def intersected_wave(state):
            # Project onto common stable states
            state1 = self.psi(state)
            state2 = other.psi(state)
            return (state1 + state2) / 2  # Average in state space

        def combined_energy(state):
            # Energy must be conserved in both contracts
            return max(self.energy(state), other.energy(state))

        # Combine transforms that preserve both contracts
        common_transforms = {T for T in self.transforms if T in other.transforms}

        return QuantumContract(
            self.agent1,  # Take primary agent from first contract
            other.agent2,  # and secondary from second
            intersected_wave,
            combined_energy,
            common_transforms,
        )

    def model(self, other):
        """Model another contract's behavior (including self)."""

        def modeling_wave(state):
            # Contract's prediction of other contract's behavior
            predicted_state = self.psi(state)
            actual_state = other.psi(state)
            return predicted_state  # Our model's prediction

        def modeling_energy(state):
            # Energy cost of modeling = difference between prediction and reality
            predicted = self.psi(state)
            actual = other.psi(state)
            return np.sum(abs(predicted - actual) ** 2)

        return QuantumContract(
            self.agent1, other.agent2, modeling_wave, modeling_energy, self.transforms
        )

    def information_flow(self, other):
        """Measure information flow between contracts."""
        # Create models in both directions
        forward_model = self.model(other)
        backward_model = other.model(self)

        # Measure prediction accuracy in both directions
        forward_error = forward_model.energy(self.psi)
        backward_error = backward_model.energy(other.psi)

        return {
            "forward": forward_error,
            "backward": backward_error,
            "symmetric": np.isclose(forward_error, backward_error),
            "total_flow": -(
                forward_error + backward_error
            ),  # Higher means better exchange
        }

    def self_model(self, depth=1):
        """Create a self-modeling contract of specified depth."""
        if depth < 1:
            return self

        # Base case: direct self-modeling
        if depth == 1:
            return self.model(self)

        # Recursive case: model self modeling self...
        inner_model = self.self_model(depth - 1)
        return self.model(inner_model) 