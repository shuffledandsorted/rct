"""Dialog system for completing partial commands with understanding.

This dialog is created when a command is understood but incomplete.
It treats arguments as quantum states, allowing for:
- Superposition of multiple suggestions
- Interference from repeated terms
- Entanglement between parameters
- Simultaneous collapse of related dimensions
"""

from typing import Dict, Any, Optional, List, Set
import numpy as np
from .dialog import SelfDiscoveringDialog, command, default


class UnderstandingDialog(SelfDiscoveringDialog):
    """A dialog that handles command completion through quantum-inspired patterns.

    Arguments can be specified in various quantum-like ways:
    - Multiple terms create superpositions: "quantum temporal"
    - Repeated terms cause interference: "quantum quantum quantum"
    - Related terms suggest entanglement: "quantum" affects temperature
    - Clear patterns collapse multiple parameters at once
    """

    def __init__(
        self,
        parent: SelfDiscoveringDialog,
        command: str,
        missing_params: Dict[str, type],
    ):
        super().__init__()
        self.parent = parent
        self.command = command
        self.missing = missing_params
        self.collected: Dict[str, Any] = {}
        # Track quantum state of arguments
        self.param_states: Dict[str, Dict[str, float]] = {
            param: {} for param in missing_params
        }

    def _process_quantum_input(self, words: List[str]) -> Dict[str, float]:
        """Process input words into quantum amplitudes."""
        amplitudes = {}
        # Count repetitions for interference
        for word in words:
            if word in amplitudes:
                # Constructive interference
                amplitudes[word] *= 1.2
            else:
                amplitudes[word] = 1.0

        # Normalize
        total = sum(x * x for x in amplitudes.values())
        if total > 0:
            factor = 1.0 / np.sqrt(total)
            amplitudes = {k: v * factor for k, v in amplitudes.items()}

        return amplitudes

    def _find_matching_params(
        self, amplitudes: Dict[str, float]
    ) -> Dict[str, Set[str]]:
        """Find which parameters the input might match."""
        matches = {param: set() for param in self.missing}

        for word, amplitude in amplitudes.items():
            # Check each parameter's suggestions
            for param in self.missing:
                suggest_method = f"_suggest_{param}"
                if hasattr(self.parent, suggest_method):
                    suggestions = getattr(self.parent, suggest_method)()
                    # If word is similar to any suggestion
                    for suggestion in suggestions:
                        if word.lower() in suggestion.lower():
                            matches[param].add(suggestion)
                            # Update quantum state
                            if suggestion not in self.param_states[param]:
                                self.param_states[param][suggestion] = 0
                            self.param_states[param][suggestion] += amplitude

        return matches

    def _try_collapse_states(self) -> Dict[str, Any]:
        """Try to collapse quantum states into definite values."""
        collapsed = {}

        for param, states in self.param_states.items():
            if not states:
                continue

            # Find highest amplitude state
            max_amp = max(states.values())
            candidates = [v for v, amp in states.items() if amp >= max_amp * 0.9]

            if len(candidates) == 1:
                # Clear collapse to one value
                value = candidates[0]
                try:
                    # Convert to parameter's type
                    typed_value = self.missing[param](value)
                    collapsed[param] = typed_value
                except (ValueError, TypeError):
                    continue

        return collapsed

    @default
    def complete(self, user_input: str) -> Optional[SelfDiscoveringDialog]:
        """Process user input as quantum-like argument patterns."""
        if user_input.lower() in ["cancel", "quit", "exit"]:
            print("\nReturning to main dialog...")
            return self.parent

        # Process input into quantum amplitudes
        words = user_input.split()
        amplitudes = self._process_quantum_input(words)

        # Find potential parameter matches
        matches = self._find_matching_params(amplitudes)

        # Try to collapse to definite values
        collapsed = self._try_collapse_states()

        # Update collected values
        self.collected.update(collapsed)
        for param in collapsed:
            self.missing.pop(param)

        # If we've collected everything
        if not self.missing:
            print("\nI understand! Executing completed command...")
            method = getattr(self.parent, self.command)
            return method(**self.collected)

        # Show the quantum state and what's still needed
        self._show_quantum_state()
        return self

    def _show_quantum_state(self):
        """Show the current quantum state of parameters."""
        print("\nCurrent understanding:")

        # Show collected values
        if self.collected:
            print("Collapsed parameters:")
            for param, value in self.collected.items():
                print(f'  {param} = "{value}"')

        # Show quantum states
        print("\nQuantum states:")
        for param in self.missing:
            states = self.param_states[param]
            if states:
                print(f"  {param}:")
                # Sort by amplitude
                sorted_states = sorted(states.items(), key=lambda x: x[1], reverse=True)
                for value, amplitude in sorted_states[:3]:  # Show top 3
                    print(f'    "{value}" ({amplitude:.2f} amplitude)')
            else:
                print(f"  {param}: no quantum state yet")

        # Show next step
        print("\nYou can:")
        print("  - Provide more terms to influence the quantum states")
        print("  - Repeat terms to increase their amplitude")
        print("  - Type 'cancel' to return to main dialog")

    @command
    def help(self) -> Optional[SelfDiscoveringDialog]:
        """Show help for completing the command."""
        print(f"\nI'm helping you complete the '{self.command}' command.")
        print("Missing information:")
        for param, param_type in self.missing.items():
            print(f"  {param} ({param_type.__name__})")
        print("\nYou can:")
        print("  - Provide more terms to influence the quantum states")
        print("  - Repeat terms to increase their amplitude")
        print("  - Type 'cancel' to return to main dialog")
        return self
