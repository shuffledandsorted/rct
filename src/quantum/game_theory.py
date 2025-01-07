import numpy as np
from typing import List, Tuple, Dict
from .utils import quantum_normalize, calculate_geodesic_collapse

def calculate_nash_payoff(state1: np.ndarray, state2: np.ndarray) -> float:
    """Calculate Nash equilibrium payoff between quantum states."""
    # Quantum game payoff using state overlap
    overlap = np.abs(np.sum(np.conj(state1) * state2))
    return float(overlap ** 2)  # Use probability as payoff

def find_pareto_optimal(states: List[np.ndarray]) -> List[int]:
    """Find Pareto optimal quantum states."""
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
    """Calculate Nash bargaining solution between quantum states."""
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
    """Calculate cooperative solution for multiple quantum states."""
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