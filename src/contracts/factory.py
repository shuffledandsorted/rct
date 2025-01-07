import numpy as np
import scipy.linalg
from typing import Set, Callable
from .base import QuantumContract
from .utils import tensor_product


def create_measurement_contract(agent1, agent2, **kwargs):
    """Create a contract for quantum measurement interaction."""
    basis = kwargs.get("basis", agent1.basis)
    
    def measurement_wave(state):
        # Project onto measurement basis
        return np.dot(basis.T.conj(), np.dot(basis, state))
        
    def energy_measure(state):
        # Von Neumann entropy
        rho = np.outer(state, state.conj())
        eigenvals = np.linalg.eigvalsh(rho)
        return -np.sum(eigenvals * np.log2(eigenvals + 1e-10))
        
    transforms: Set[Callable] = {
        lambda psi: psi / (np.sqrt(np.sum(np.abs(psi) ** 2)) + 1e-10),
        lambda psi: np.dot(basis.T.conj(), psi),
    }
    
    return QuantumContract(agent1, agent2, measurement_wave, energy_measure, transforms)


def create_evolution_contract(agent1, agent2, **kwargs):
    """Create a contract for quantum evolution interaction."""
    hamiltonian = kwargs.get("hamiltonian", np.eye(len(agent1.state)))
    dt = kwargs.get("dt", 0.1)
    
    def evolution_wave(state):
        # Time evolution operator
        U = scipy.linalg.expm(-1j * hamiltonian * dt)
        return np.dot(U, state)
        
    def energy_measure(state):
        # Expected energy
        return np.real(np.dot(state.conj(), np.dot(hamiltonian, state)))
        
    transforms: Set[Callable] = {
        lambda psi: psi / (np.sqrt(np.sum(np.abs(psi) ** 2)) + 1e-10),
        lambda psi: scipy.linalg.expm(-1j * hamiltonian * dt) @ psi,
    }
    
    return QuantumContract(agent1, agent2, evolution_wave, energy_measure, transforms)


def create_entanglement_contract(agent1, agent2, **kwargs):
    """Create a contract for quantum entanglement interaction."""
    
    def entangle_wave(state):
        # Create entangled state
        psi = tensor_product(agent1.state, agent2.state)
        return (psi + np.roll(psi, 1)) / np.sqrt(2)
        
    def energy_measure(state):
        # Entanglement entropy
        rho = np.outer(state, state.conj())
        reduced = np.trace(rho.reshape(2, 2, 2, 2), axis1=1, axis2=3)
        eigenvals = np.linalg.eigvalsh(reduced)
        return -np.sum(eigenvals * np.log2(eigenvals + 1e-10))
        
    transforms: Set[Callable] = {
        lambda psi: psi / (np.sqrt(np.sum(np.abs(psi) ** 2)) + 1e-10),
        lambda psi: (psi + np.roll(psi, 1)) / np.sqrt(2),
    }
    
    return QuantumContract(agent1, agent2, entangle_wave, energy_measure, transforms)


def create_negotiation_contract(agent1, agent2, **kwargs):
    """Create a contract specifically for negotiation between agents."""
    tolerance = kwargs.get("tolerance", 1e-6)
    
    def negotiate_wave(state):
        # Combine states with phase alignment
        phase1 = np.angle(agent1.state)
        phase2 = np.angle(agent2.state)
        phase_diff = np.exp(1j * (phase1 - phase2))
        return (state + phase_diff * agent2.state) / np.sqrt(2)
        
    def energy_measure(state):
        # Measure agreement through state overlap
        overlap = np.abs(np.dot(state.conj(), agent2.state)) ** 2
        return -np.log(overlap + 1e-10)
        
    transforms: Set[Callable] = {
        lambda psi: psi / (np.sqrt(np.sum(np.abs(psi) ** 2)) + 1e-10),
        lambda psi: (psi + np.roll(psi, 1)) / np.sqrt(2),
        lambda psi: np.exp(1j * np.angle(psi)),  # Phase alignment
    }
    
    contract = QuantumContract(agent1, agent2, negotiate_wave, energy_measure, transforms)
    
    # Immediately try to reach agreement
    if not contract.negotiate(tolerance=tolerance):
        raise ValueError("Failed to reach negotiation agreement")
        
    return contract 