"""Self-discovering command-line dialog system.

Commands marked with @command appear in the help menu.
All commands support argument parsing and state transitions.

The system acts as a contract negotiator between commands and users:
1. Commands define their requirements (parameters, types)
2. Users express their intent (partial or complete commands)
3. The dialog system mediates understanding through:
   - Template-based negotiation of missing parameters
   - Suggestions drawn from context
   - Gradual collapse into a fully specified command

This creates a natural flow where incomplete commands become
opportunities for negotiating shared understanding.
"""

from typing import Dict, Any, Optional, Callable, TypeVar, cast, Tuple, Type
import inspect
import asyncio
import shlex
import sys
from functools import wraps

F = TypeVar("F", bound=Callable[..., Any])


# Decorators
def command(f: F) -> F:
    """Mark a method to appear in the help menu."""
    setattr(f, "_is_command", True)
    return f


def default(f: F) -> F:
    """Mark a method as the default command handler."""
    setattr(f, "_is_default", True)
    return f


def exit_command(f: F) -> F:
    """Mark a method as the exit command handler."""
    setattr(f, "_is_exit", True)
    return f


# Argument parsing
def _parse_args(f: F, raw_args: list) -> dict:
    """Parse arguments according to function signature."""
    sig = inspect.signature(f)
    params = list(sig.parameters.values())[1:]  # Skip 'self'
    call_args = {}
    missing_params = {}

    for i, param in enumerate(params):
        if param.kind == param.VAR_POSITIONAL:
            call_args[param.name] = raw_args[i:]
            break
        elif i < len(raw_args):
            call_args[param.name] = raw_args[i]
        elif param.default != param.empty:
            continue
        else:
            # Track missing required params with their types
            missing_params[param.name] = (
                param.annotation if param.annotation != param.empty else str
            )

    if missing_params:
        raise ValueError("_missing_params:" + str(missing_params))
    return call_args

class SelfDiscoveringDialog:
    def __init__(self):
        self._commands = self._discover_commands()
        self._default = self._discover_default()
        self._help_text = self._make_help_text()

    # Command discovery and help
    def _discover_commands(self) -> Dict[str, Dict[str, Any]]:
        """Find methods marked with @command for the help menu."""
        commands = {}
        for name, method in inspect.getmembers(self.__class__, inspect.isfunction):
            if hasattr(method, "_is_command") and method.__doc__:
                doc = method.__doc__.split("\n")[0]
                sig = inspect.signature(method)
                commands[name.lower()] = {
                    "method": name,
                    "help": doc,
                    "signature": sig,
                }
        return commands

    def _discover_default(self) -> Callable[[str], None]:
        """Find the @default handler or use fallback."""
        for _, method in inspect.getmembers(self.__class__, inspect.isfunction):
            if hasattr(method, "_is_default"):
                return method
        return self._fallback_command

    def _make_help_text(self) -> str:
        """Generate help text from @command methods."""
        lines = ["Available commands:"]
        for name, info in self._commands.items():
            sig = info["signature"]
            params = [p for p in sig.parameters if p != "self"]
            if params:
                lines.append(f"  {name} {' '.join(params)}: {info['help']}")
            else:
                lines.append(f"  {name}: {info['help']}")
        return "\n".join(lines)

    # Input processing
    def _tokenize(self, user_input: str) -> list[str]:
        """Try different strategies to tokenize user input."""
        try:
            return shlex.split(user_input)
        except ValueError:
            return user_input.split()

    def _find_closest_command(self, tokens: list[str]) -> Tuple[Callable, list]:
        """Find closest matching command and prepare its arguments.
        Returns (method, args) tuple ready for execution."""
        # Handle empty input
        if not tokens:
            return self._default, []

        # Try to find matching command
        cmd = tokens[0].lower()
        for command in self._commands:
            if command.startswith(cmd):
                method = getattr(self, self._commands[command]["method"])
                return method, tokens[1:]

        # Fall back to default
        return self._default, tokens

    def process_command(self, user_input: str) -> Optional["SelfDiscoveringDialog"]:
        """Process a command and return the next dialog state."""
        tokens = self._tokenize(user_input)
        method, args = self._find_closest_command(tokens)

        # Execute
        try:
            call_args = _parse_args(method, args)
            bound_method = method.__get__(self, self.__class__)
            result = bound_method(**call_args)
            # Commands with @exit_command return None to exit
            return None if hasattr(method, "_is_exit") else self
        except ValueError as e:
            err_msg = str(e)
            if err_msg.startswith("_missing_params:"):
                # Create understanding dialog for missing params
                from .understanding_dialog import UnderstandingDialog

                missing_str = err_msg.split(":", 1)[1]
                # Safely eval the dict string since it contains type objects
                missing_params = eval(missing_str)
                return UnderstandingDialog(
                    parent=self, command=method.__name__, missing_params=missing_params
                )
            print(f"Error: {err_msg}")
            return self
        except Exception as e:
            print(f"Error executing {method.__name__}: {e}")
            return self

    # Built-in commands
    @command
    def help(self) -> None:
        """Display available commands."""
        print(self._help_text)

    @exit_command
    def exit(self) -> None:
        """Exit the dialog system."""

    def _fallback_command(self, user_input: str = "") -> None:
        """Default fallback when no command matches or no input received."""
        if not user_input:
            print("No input received. Type 'help' for available commands.")
        else:
            print(f"Unknown command. Type 'help' for available commands.")


async def async_dialog(
    dialog: SelfDiscoveringDialog, prompt: str = ""
) -> Optional[SelfDiscoveringDialog]:
    """Get user input asynchronously and process it through the dialog system."""
    try:
        # Get user input (use asyncio.to_thread to avoid blocking)
        user_input = await asyncio.to_thread(input, prompt)

        # Process the input through dialog system
        return dialog.process_command(user_input)
    except (EOFError, KeyboardInterrupt):
        return None
