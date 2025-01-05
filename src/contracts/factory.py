import numpy as np
import scipy.linalg

from .base import QuantumContract
from .utils import tensor_product


def create_measurement_contract(agent1, agent2, **kwargs):
    """Create a contract for quantum measurement interaction."""

    def measurement_wave(state):
        return np.dot(agent1.basis, state)

    def energy_measure(state):
        return -np.log(abs(state) ** 2 + 1e-10)

    transforms = {
        lambda psi: psi / np.sqrt(np.sum(abs(psi) ** 2)),
        lambda psi: np.conjugate(psi),
    }

    return QuantumContract(agent1, agent2, measurement_wave, energy_measure, transforms)


def create_evolution_contract(agent1, agent2, **kwargs):
    """Create a contract for quantum evolution interaction."""
    hamiltonian = kwargs.get("hamiltonian", np.eye(len(agent1.state)))

    def evolution_wave(state):
        return np.exp(-1j * hamiltonian) @ state

    def energy_measure(state):
        return np.real(np.dot(np.conjugate(state), hamiltonian @ state))

    transforms = {
        lambda psi: scipy.linalg.expm(-1j * hamiltonian) @ psi,
        lambda psi: psi / np.sqrt(np.sum(abs(psi) ** 2)),
    }

    return QuantumContract(agent1, agent2, evolution_wave, energy_measure, transforms)


def create_entanglement_contract(agent1, agent2, **kwargs):
    """Create a contract for quantum entanglement interaction."""

    def entangle_wave(state):
        return (
            tensor_product(agent1.state, agent2.state)
            + tensor_product(agent2.state, agent1.state)
        ) / np.sqrt(2)

    def energy_measure(state):
        rho = np.outer(state, np.conjugate(state))
        eigenvals = np.linalg.eigvalsh(rho)
        return -np.sum(eigenvals * np.log2(eigenvals + 1e-10))

    transforms = {
        lambda psi: psi / np.sqrt(np.sum(abs(psi) ** 2)),
        lambda psi: np.conjugate(psi),
    }

    return QuantumContract(agent1, agent2, entangle_wave, energy_measure, transforms) 