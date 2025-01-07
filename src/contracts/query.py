import numpy as np
from typing import Optional, Dict, Any

from .base import QuantumContract
from ..quantum.utils import text_to_quantum_pattern, calculate_cohesion


class QueryContract(QuantumContract):
    """Contract for stable information exchange between user and knowledge base.

    This contract enables querying a knowledge base while maintaining system stability through:
    1. Energy conservation during information exchange
    2. Symmetric participation between user and knowledge base
    3. Fixed point convergence for stable results
    4. Coherence checks for meaningful information flow
    """

    def __init__(self, user_agent, knowledge_agent, coherence_threshold: float = 0.7):
        """Initialize query contract between user and knowledge agents.

        Args:
            user_agent: Agent representing the user/query source
            knowledge_agent: Agent representing the knowledge base
            coherence_threshold: Minimum coherence required for valid knowledge extraction
        """

        def query_wave_fn(state):
            """Wave function for query state evolution.

            Combines user intent with knowledge state while preserving energy.
            """
            # Get current knowledge state
            knowledge_state = knowledge_agent.state

            # Combine states with phase alignment for coherent exchange
            phase_diff = np.exp(1j * (np.angle(state) - np.angle(knowledge_state)))
            combined = state + phase_diff * knowledge_state

            # Normalize while preserving relative amplitudes
            return combined / np.sqrt(2)

        def query_energy_fn(state):
            """Energy function measuring query-knowledge alignment.

            Lower energy indicates better alignment between query and knowledge.
            """
            # Calculate overlap between query and knowledge states
            knowledge_state = knowledge_agent.state
            overlap = np.abs(np.vdot(state, knowledge_state)) ** 2

            # Convert to energy (lower is better)
            return -np.log(overlap + 1e-10)

        # Define allowed transformations that preserve stability
        transforms = {
            # Normalize while preserving relative phases
            lambda psi: psi / (np.sqrt(np.sum(np.abs(psi) ** 2)) + 1e-10),
            # Phase alignment for coherent information exchange
            lambda psi: np.exp(1j * np.angle(psi)),
        }

        super().__init__(
            agent1=user_agent,
            agent2=knowledge_agent,
            wave_fn=query_wave_fn,
            energy_fn=query_energy_fn,
            transforms=transforms,
        )

        self.coherence_threshold = coherence_threshold

    def negotiate(self, max_iterations: int = 100, tolerance: float = 1e-6) -> bool:
        """Implement recursive contract negotiation to find fixed point.

        Args:
            max_iterations: Maximum number of iterations to try
            tolerance: Convergence tolerance for fixed point

        Returns:
            True if negotiation reached stable state, False otherwise
        """
        # Get initial states
        state = self.agent1.wave_fn.amplitude
        prev_state = None
        iteration = 0

        while iteration < max_iterations:
            # Apply contract transformations
            new_state = self.psi(state)
            for transform in self.transforms:
                new_state = transform(new_state)

            # Check for fixed point convergence
            if prev_state is not None:
                diff = np.linalg.norm(new_state - prev_state)
                if diff < tolerance:
                    self._fixed_point = new_state
                    return True

            prev_state = new_state
            state = new_state
            iteration += 1

        return False

    def process_query(self, query_text: str) -> Optional[Dict[str, Any]]:
        """Process a query while maintaining system stability.

        Args:
            query_text: The text of the query to process

        Returns:
            Dictionary containing query results if successful, None otherwise
        """
        # Convert query text to quantum state
        query_state = text_to_quantum_pattern(query_text, self.agent1.dims)

        # Set initial state and try to reach stability
        self.agent1.wave_fn.amplitude = query_state
        if not self.negotiate():
            return None

        # Extract knowledge from stable state
        if self._fixed_point is not None:
            return self._extract_knowledge(self._fixed_point)
        return None

    def _extract_knowledge(self, state: np.ndarray) -> Optional[Dict[str, Any]]:
        """Extract knowledge from stabilized quantum state.

        Args:
            state: The stabilized quantum state to extract knowledge from

        Returns:
            Dictionary containing extracted knowledge if coherence is sufficient
        """
        # Combine states of both agents
        user_state = self.agent1.state
        knowledge_state = self.agent2.state
        combined_state = (user_state + knowledge_state) / np.sqrt(
            2
        )  # Normalized combination

        # Calculate coherence using existing function
        coherence = calculate_cohesion([state, combined_state])

        # Only return results if coherence meets threshold
        if coherence >= self.coherence_threshold:
            # Get knowledge from agent
            knowledge = self.agent2.get_knowledge(state)

            # Add metadata about the exchange
            return {
                "knowledge": knowledge,
                "coherence": coherence,
                "energy": self.energy(state),
            }

        return None
