from .utils import Y


def form_collective_reality(contracts):
    """Form collective reality from a set of contracts."""
    if not contracts:
        raise ValueError("Need at least one contract")

    # Start with first contract
    reality = contracts[0]

    # Intersect with remaining contracts
    for contract in contracts[1:]:
        reality = reality.intersect(contract)

    # Find fixed point of collective reality
    def reality_evolution(f):
        def evolve(state):
            return reality.psi(f(state))

        return evolve

    try:
        # Fixed point is the collective reality
        collective_state = Y(reality_evolution)(reality.psi)
        return collective_state
    except RecursionError:
        raise ValueError("No stable collective reality found") 