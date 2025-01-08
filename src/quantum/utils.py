"""Quantum mechanical utility functions for RCT operations.

This module provides core quantum mechanical operations used throughout RCT:
- State normalization and energy conservation
- Geodesic paths between states
- Coherence and cohesion metrics
- Pattern encoding and transformation

These utilities maintain quantum mechanical properties while providing
practical tools for quantum state manipulation and measurement.
"""

import numpy as np
from typing import List, Optional

def quantum_normalize(state: np.ndarray, min_energy: float = 1e-10) -> np.ndarray:
    """Normalize quantum state while preserving energy levels and phase relationships.

    Performs quantum normalization that:
    1. Preserves relative phase relationships between components
    2. Maintains minimum energy levels to prevent collapse
    3. Ensures total probability sums to 1

    The process handles very small energy states by introducing a minimum
    energy floor, preventing numerical instability while maintaining
    quantum mechanical properties.

    Args:
        state: Complex quantum state to normalize
        min_energy: Minimum total energy to maintain

    Returns:
        Normalized quantum state with preserved structure
    """
    prob_amplitudes = np.abs(state) ** 2
    total_energy = np.sum(prob_amplitudes)

    if total_energy < min_energy:
        base_energy = min_energy / state.size
        min_amplitudes = np.full_like(prob_amplitudes, base_energy)
        phases = np.angle(state)
        return np.sqrt(min_amplitudes) * np.exp(1j * phases)

    return state / np.sqrt(total_energy)

def calculate_geodesic_collapse(state1: np.ndarray, state2: np.ndarray, t: float = 0.5) -> np.ndarray:
    """Calculate geodesic path between quantum states and collapse to point t.

    Finds the shortest path between quantum states in their Hilbert space
    and collapses to a point along that path. The process:
    1. Aligns phases between states to enable smooth interpolation
    2. Calculates geodesic angle in state space
    3. Interpolates along great circle path

    This maintains quantum mechanical properties while providing optimal
    interpolation between states.

    Args:
        state1: Starting quantum state
        state2: Ending quantum state
        t: Interpolation parameter (0 = state1, 1 = state2)

    Returns:
        Intermediate quantum state along geodesic path
    """
    # First align phases between states
    overlap = np.sum(np.conj(state1) * state2)
    phase_factor = np.exp(1j * np.angle(overlap))
    aligned_state2 = state2 * phase_factor

    # Calculate geodesic angle
    cos_theta = np.real(np.sum(np.conj(state1) * aligned_state2))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    # If states are effectively identical, return first state
    if np.abs(theta) < 1e-10:
        return state1

    # Calculate geodesic path
    sin_theta = np.sin(theta)
    if sin_theta < 1e-10:
        return state1

    # Interpolate along geodesic while preserving phase
    collapsed = (np.sin((1-t)*theta) * state1 + np.sin(t*theta) * aligned_state2) / sin_theta

    return quantum_normalize(collapsed)

def calculate_density_matrix(state: np.ndarray) -> np.ndarray:
    """Calculate density matrix from quantum state.

    Constructs the density matrix ρ = |ψ⟩⟨ψ| which represents the
    quantum state in operator form. This is useful for:
    - Calculating expectation values
    - Performing partial traces
    - Analyzing mixed states

    Args:
        state: Pure quantum state vector

    Returns:
        Density matrix representation
    """
    return np.outer(state, np.conj(state))

def calculate_coherence(state: np.ndarray) -> float:
    """Calculate quantum coherence using relative phase coherence and amplitude distribution.

    This function measures the internal coherence of a single quantum state through:
    1. Phase coherence - how aligned the phases are between components
    2. Amplitude distribution - how non-uniform the amplitudes are

    The final coherence score is weighted:
    - 70% from phase coherence (normalized to [0,1])
    - 30% from amplitude distribution (penalizing uniform distributions)

    This weighting emphasizes the importance of phase relationships while still
    accounting for amplitude structure.

    Args:
        state: Quantum state to calculate coherence for

    Returns:
        Float between 0 and 1 representing internal coherence
    """
    # Phase coherence
    phases = np.angle(state)
    phase_diffs = phases.reshape(-1, 1) - phases.reshape(1, -1)
    phase_coherence = float(np.mean(np.cos(phase_diffs)))

    # Amplitude distribution (penalize uniform distributions)
    amplitudes = np.abs(state)
    amplitude_var = np.var(amplitudes)
    amplitude_score = 1.0 - np.exp(-5 * amplitude_var)  # Exponential scaling

    # Combine scores with emphasis on phase coherence
    return float(0.7 * (phase_coherence + 1) / 2 + 0.3 * amplitude_score)

def calculate_cohesion(states: List[np.ndarray]) -> float:
    """Calculate quantum cohesion using phase alignment and state distinctness.

    This function balances two key aspects:
    1. Quantum overlap between state pairs - measures alignment and agreement
    2. State distinctness - ensures meaningful differences between states

    The final cohesion score is weighted:
    - 60% from average pairwise overlap (alignment)
    - 40% from distinctness factor (unique contributions)

    This balance helps prevent "echo chamber" scenarios where states simply
    mirror each other, instead favoring interactions where each state
    contributes unique information while maintaining sufficient alignment.

    Args:
        states: List of quantum states to calculate cohesion between

    Returns:
        Float between 0 and 1 representing cohesion, where higher values
        indicate better balance of alignment and distinctness
    """
    if not states or len(states) == 1:
        return 0.0

    cohesion = 0.0
    n_pairs = 0
    distinctness = 0.0

    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            # Calculate quantum overlap
            overlap = np.abs(np.sum(np.conj(states[i]) * states[j]))
            overlap /= np.sqrt(np.sum(np.abs(states[i]) ** 2) * np.sum(np.abs(states[j]) ** 2))

            # Calculate state distinctness
            diff_state = states[i] - states[j]
            distinctness += np.sqrt(np.sum(np.abs(diff_state) ** 2))

            cohesion += overlap
            n_pairs += 1

    if n_pairs == 0:
        return 0.0

    # Normalize distinctness
    distinctness = distinctness / (n_pairs * np.sqrt(2))  # √2 is max difference for normalized states

    # Combine overlap cohesion with distinctness
    avg_cohesion = cohesion / n_pairs
    distinctness_factor = 1.0 - np.exp(-2 * distinctness)  # Exponential scaling

    return float(0.6 * avg_cohesion + 0.4 * distinctness_factor)

def text_to_quantum_pattern(text: str, dims: tuple) -> np.ndarray:
    """Convert text to quantum pattern using RCT encoding principles.

    Transforms text into a quantum state representation that preserves:
    1. Character relationships through phase encoding
    2. Spatial structure through 2D reshaping
    3. Frequency patterns through Fourier transformation

    The process:
    1. Convert characters to complex amplitudes
    2. Encode position information in phases
    3. Create 2D quantum state with FFT
    4. Normalize while preserving structure

    Args:
        text: String to encode
        dims: Tuple of (height, width) for 2D representation

    Returns:
        Quantum state encoding the text pattern
    """
    # Convert text to complex amplitudes
    chars = np.array([ord(c) for c in text], dtype=np.complex128)

    # Apply RCT phase encoding
    positions = np.arange(len(chars))
    phases = 2j * np.pi * positions / len(chars)
    chars *= np.exp(phases)

    # Create quantum state representation
    size = dims[0] * dims[1]
    if len(chars) < size:
        chars = np.pad(chars, (0, size - len(chars)))
    else:
        chars = chars[:size]

    # Reshape and apply FFT for phase relationships
    pattern = np.fft.fft2(chars.reshape(dims))

    return quantum_normalize(pattern) 
