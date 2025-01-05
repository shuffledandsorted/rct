from typing import List, Any


class ContractAgent:
    """Mixin class for agents that can interact through contracts."""

    @property
    def state(self):
        """The quantum state of the agent."""
        raise NotImplementedError(
            "Classes using ContractAgent must implement state property"
        )

    def __init__(self):
        self._contracts = []

    def get_active_contracts(self):
        """Get list of active contracts involving this agent."""
        return [c for c in self._contracts if c.is_valid()]

    def compose_contracts(self, agent_path):
        """Compose a chain of contracts along a path of agents."""
        if len(agent_path) < 2:
            raise ValueError("Need at least two agents to compose contracts")

        # Find contracts connecting consecutive agents
        contracts = []
        for i in range(len(agent_path) - 1):
            matching = [
                c
                for c in self._contracts
                if (c.agent1 == agent_path[i] and c.agent2 == agent_path[i + 1])
                or (c.agent2 == agent_path[i] and c.agent1 == agent_path[i + 1])
            ]
            if not matching:
                raise ValueError(
                    f"No contract between {agent_path[i]} and {agent_path[i+1]}"
                )
            contracts.append(matching[0])

        # Compose all contracts
        result = contracts[0]
        for contract in contracts[1:]:
            result = result.compose(contract)
        return result 