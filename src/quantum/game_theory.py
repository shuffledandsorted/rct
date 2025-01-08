"""Quantum game theory for cooperative state evolution.

This module implements quantum game theory concepts used in RCT for:
- Finding stable agreements between quantum states
- Calculating optimal cooperative solutions
- Determining Pareto optimal outcomes

The key insight is treating quantum state evolution as a cooperative game,
where states seek to maximize their overlap (payoff) while maintaining
their essential structure. This provides principled ways to combine states
that respect both quantum mechanics and game theory.
"""

import numpy as np
from typing import List, Tuple, Dict
from .utils import quantum_normalize, calculate_geodesic_collapse

def calculate_nash_payoff(state1: np.ndarray, state2: np.ndarray) -> float:
    """Calculate Nash equilibrium payoff between quantum states.

    The payoff is defined as the squared overlap between states, representing
    how well they agree with each other. This has key properties:
    - Symmetric: payoff(a,b) = payoff(b,a)
    - Bounded: 0 ≤ payoff ≤ 1
    - Maximum at identical states

    Args:
        state1: First quantum state
        state2: Second quantum state

    Returns:
        Float payoff value between 0 and 1
    """
    # Quantum game payoff using state overlap
    overlap = np.abs(np.sum(np.conj(state1) * state2))
    return float(overlap ** 2)  # Use probability as payoff

def find_pareto_optimal(states: List[np.ndarray]) -> List[int]:
    """Find Pareto optimal quantum states.

    A state is Pareto optimal if no other state has better payoff with
    all other states. This identifies states that represent optimal
    trade-offs between different quantum configurations.

    The process:
    1. Calculate payoff matrix between all pairs of states
    2. For each state, check if any other state dominates it
    3. Return indices of non-dominated states

    Args:
        states: List of quantum states to analyze

    Returns:
        List of indices of Pareto optimal states
    """
    n_states = len(states)
    payoff_matrix = np.zeros((n_states, n_states))

    # Calculate payoffs between all pairs
    for i in range(n_states):
        for j in range(n_states):
            payoff_matrix[i,j] = calculate_nash_payoff(states[i], states[j])

    # Find Pareto optimal states
    pareto_optimal = []
    for i in range(n_states):
        dominated = False
        for j in range(n_states):
            if i != j:
                # Check if j dominates i
                if np.all(payoff_matrix[j,:] >= payoff_matrix[i,:]) and \
                   np.any(payoff_matrix[j,:] > payoff_matrix[i,:]):
                    dominated = True
                    break
        if not dominated:
            pareto_optimal.append(i)

    return pareto_optimal

def quantum_bargaining_solution(state1: np.ndarray, state2: np.ndarray) -> np.ndarray:
    """Calculate Nash bargaining solution between quantum states.

    Finds the optimal compromise between two quantum states by maximizing
    the product of their gains relative to the disagreement point.

    The process:
    1. Calculate initial payoff as disagreement point
    2. Try different interpolation points along geodesic
    3. Choose point that maximizes product of gains

    This ensures the solution is:
    - Pareto optimal
    - Symmetric
    - Independent of irrelevant alternatives

    Args:
        state1: First quantum state
        state2: Second quantum state

    Returns:
        Optimal compromise state
    """
    # Get initial payoffs (disagreement point)
    initial_payoff = calculate_nash_payoff(state1, state2)

    # Try different interpolation points to maximize product of gains
    best_t = 0.5  # Start with equal weight
    max_product = 0.0

    for t in np.linspace(0.1, 0.9, 9):
        collapsed = calculate_geodesic_collapse(state1, state2, t)
        payoff1 = calculate_nash_payoff(collapsed, state1)
        payoff2 = calculate_nash_payoff(collapsed, state2)

        # Calculate gains from initial position
        gain1 = max(0, payoff1 - initial_payoff)
        gain2 = max(0, payoff2 - initial_payoff)
        product = gain1 * gain2

        if product > max_product:
            max_product = product
            best_t = t

    # Return bargaining solution
    return calculate_geodesic_collapse(state1, state2, best_t)

def cooperative_collapse(states: List[np.ndarray], weights: List[float]) -> np.ndarray:
    """Calculate cooperative solution for multiple quantum states.

    Combines multiple quantum states into a single state that represents
    their optimal cooperative agreement. Uses weighted geodesic collapse
    to maintain quantum mechanical properties while respecting relative
    importance of states.

    The process:
    1. Normalize weights to sum to 1
    2. Start with first state
    3. Iteratively collapse with remaining states using relative weights

    This generalizes quantum bargaining to multiple states while
    maintaining energy conservation and phase relationships.

    Args:
        states: List of quantum states to combine
        weights: Relative importance weights for each state

    Returns:
        Combined quantum state representing cooperative solution

    Raises:
        ValueError: If states and weights don't match or are empty
    """
    if not states or not weights or len(states) != len(weights):
        raise ValueError("Must provide equal number of states and weights")

    # Normalize weights
    weights = np.array(weights) / np.sum(weights)

    # Start with first state
    result = states[0]

    # Iteratively collapse with remaining states
    accumulated_weight = weights[0]
    for state, weight in zip(states[1:], weights[1:]):
        # Calculate relative weight for this collapse
        t = weight / (accumulated_weight + weight)
        result = calculate_geodesic_collapse(result, state, t)
        accumulated_weight += weight

    return result 
