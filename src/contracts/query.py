import numpy as np
from typing import Optional, Dict, Any

from .base import QuantumContract
from ..quantum.utils import (
    text_to_quantum_pattern,
    quantum_normalize,
    calculate_cohesion,
)


class QueryContract(QuantumContract):
    """Contract for stable information exchange between user and knowledge base.

    This contract enables querying a knowledge base while maintaining system stability through:
    1. Energy conservation during information exchange
    2. Symmetric participation between user and knowledge base
    3. Natural convergence to energy eigenstates
    4. Coherence checks for meaningful information flow
    """

    def __init__(self, user_agent, knowledge_agent, coherence_threshold: float = 0.7):
        """Initialize query contract between user and knowledge agents."""

        def query_wave_fn(state):
            """Wave function for query state evolution.

            Combines user intent with knowledge state while preserving energy.
            Natural quantum dynamics will converge to energy eigenstates.
            """
            knowledge_state = knowledge_agent.state
            # Simple superposition - quantum dynamics will handle convergence
            return quantum_normalize(state + knowledge_state)

        def query_energy_fn(state):
            """Energy function defining the Hamiltonian of the system.

            The system will naturally evolve to minimize this energy.
            """
            knowledge_state = knowledge_agent.state
            # Negative overlap encourages alignment
            return -np.abs(np.vdot(state, knowledge_state)) ** 2

        super().__init__(
            agent1=user_agent,
            agent2=knowledge_agent,
            wave_fn=query_wave_fn,
            energy_fn=query_energy_fn,
            transforms=set(),  # No explicit transforms needed
        )

        self.coherence_threshold = coherence_threshold

    def process_query(self, query_text: str) -> Optional[Dict[str, Any]]:
        """Process a query through natural quantum evolution.

        Args:
            query_text: The text of the query to process

        Returns:
            Dictionary containing query results if successful, None otherwise
        """
        # Convert query text to initial quantum state
        query_state = text_to_quantum_pattern(query_text, self.agent1.dims)
        self.agent1.wave_fn.amplitude = quantum_normalize(query_state)

        # Let quantum dynamics find the ground state
        evolved_state = self.psi(self.agent1.wave_fn.amplitude)

        # Extract knowledge if coherent enough
        return self._extract_knowledge(evolved_state)

    def _extract_knowledge(self, state: np.ndarray) -> Optional[Dict[str, Any]]:
        """Extract knowledge from the evolved quantum state."""
        # Calculate coherence between query and knowledge states
        coherence = calculate_cohesion([state, self.agent2.state])

        if coherence >= self.coherence_threshold:
            knowledge = self.agent2.get_knowledge(state)
            return {
                "knowledge": knowledge,
                "coherence": coherence,
                "energy": self.energy(state),
            }

        return None
