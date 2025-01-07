"""Self-discovering command-line dialog system.

This module provides a framework for building interactive command-line interfaces that:

1. Automatically discover commands from class methods
2. Use docstrings for help text and documentation
3. Handle argument parsing and command dispatch
4. Support state transitions between different dialog modes

Key components:

- @command: Decorator for marking methods as commands
- @default: Decorator for marking the default command handler
- SelfDiscoveringDialog: Base class that provides command discovery and processing
- run_async_dialog: Async function to run the dialog system

Example:
    ```python
    class MyDialog(SelfDiscoveringDialog):
        @command
        def greet(self, name: str) -> "MyDialog":
            '''Say hello to someone.'''
            print(f"Hello, {name}!")
            return self

        @default
        def echo(self, text: str) -> "MyDialog":
            '''Echo unknown commands.'''
            print(f"You said: {text}")
            return self

    # Run the dialog
    dialog = MyDialog()
    asyncio.run(run_async_dialog(dialog))
    ```

Commands are discovered by inspecting the class's methods. Each command can:
- Have its own arguments (parsed automatically)
- Return a new dialog state (enabling state transitions)
- Use docstrings for help text
- Handle unknown commands via the @default handler
"""

from typing import Dict, Any, Optional, Callable, TypeVar, cast, Protocol
import inspect
import asyncio
import shlex
import sys
from functools import wraps

F = TypeVar("F", bound=Callable[..., Any])


def default(f: F) -> F:
    """Decorator to mark a method as the default command handler."""
    setattr(f, "_is_default", True)
    return f


def command(f: F) -> F:
    """Decorator that handles argument parsing for command methods."""
    sig = inspect.signature(f)
    params = list(sig.parameters.values())[1:]  # Skip 'self'

    @wraps(f)
    def wrapper(self: Any, *args: Any) -> Any:
        call_args = {}
        raw_args = args[0] if args else []

        try:
            # Handle different parameter patterns
            for i, param in enumerate(params):
                if i < len(raw_args):
                    # We have a value for this parameter
                    if param.kind == param.VAR_POSITIONAL:
                        # *args parameter - collect remaining arguments
                        call_args[param.name] = raw_args[i:]
                        break
                    else:
                        call_args[param.name] = raw_args[i]
                elif param.default != param.empty:
                    # Parameter has a default value
                    continue
                elif param.kind == param.VAR_POSITIONAL:
                    # *args parameter with no values
                    call_args[param.name] = []
                else:
                    # Missing required parameter
                    raise TypeError(f"Missing required argument: {param.name}")

            return f(self, **call_args)
        except Exception as e:
            print(f"Error: {e}")
            return self

    setattr(wrapper, "_is_command", True)
    setattr(wrapper, "__doc__", f.__doc__)
    setattr(wrapper, "__signature__", sig)
    return cast(F, wrapper)


class SelfDiscoveringDialog:
    """Base class for self-reflective dialog systems.

    This class provides a command system that discovers available commands by inspecting
    its own methods. Commands are derived from public methods, with help text taken
    from their docstrings.
    """

    def __init__(self):
        """Initialize the dialog system."""
        self._commands = self._discover_commands()
        self._default = self._discover_default()
        self._help_text = self._make_help_text()

    def _discover_commands(self) -> Dict[str, Dict[str, Any]]:
        """Discover available commands by inspecting class methods."""
        commands = {}
        for name, method in inspect.getmembers(self.__class__, inspect.isfunction):
            if hasattr(method, "_is_command") or (
                not name.startswith("_") and name not in ("run", "start", "stop")
            ):
                doc = inspect.getdoc(method) or ""
                sig = inspect.signature(method)
                commands[name.lower()] = {
                    "method": name,
                    "help": doc.split("\n")[0],  # First line of docstring
                    "handler": method,
                    "signature": sig,
                }
        return commands

    def _discover_default(self) -> Callable[[str], "SelfDiscoveringDialog"]:
        """Find the method marked as the default command handler."""
        for _, method in inspect.getmembers(self.__class__, inspect.isfunction):
            if hasattr(method, "_is_default"):
                return method
        return self._fallback_command

    def _make_help_text(self) -> str:
        """Generate help text from discovered commands."""
        lines = ["Available commands:"]
        for name, info in self._commands.items():
            sig = info["signature"]
            params = [p for p in sig.parameters if p != "self"]
            if params:
                lines.append(f"  {name} {' '.join(params)}: {info['help']}")
            else:
                lines.append(f"  {name}: {info['help']}")
        return "\n".join(lines)

    @command
    def help(self) -> "SelfDiscoveringDialog":
        """Display available commands and their descriptions."""
        print(self._help_text)
        return self

    @command
    def exit(self) -> None:
        """Exit the dialog system."""
        return None

    def process_command(self, user_input: str) -> Optional["SelfDiscoveringDialog"]:
        """Process a user command, dispatching to the appropriate handler.

        Returns:
            The next dialog state, or None to exit
        """
        try:
            parts = shlex.split(user_input)
        except ValueError:
            parts = user_input.split()

        if not parts:
            return self

        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in self._commands:
            method = getattr(self, self._commands[cmd]["method"])
            return method(args)
        else:
            return self._default(user_input)

    def _fallback_command(self, user_input: str) -> "SelfDiscoveringDialog":
        """Default fallback if no @default handler is defined."""
        print(f"Unknown command. Type 'help' for available commands.")
        return self


async def run_async_dialog(
    dialog: SelfDiscoveringDialog, prompt: str = "You: "
) -> None:
    """Run a dialog system asynchronously.

    Args:
        dialog: The dialog system to run
        prompt: The prompt to display

    This function handles the main input/output loop for the dialog system.
    """
    current = dialog
    try:
        while current is not None:
            try:
                print(prompt, end="", flush=True)
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                line = line.rstrip("\n")
                current = current.process_command(line)
            except Exception as e:
                print(f"Error: {e}")
    finally:
        print("\nDialog ended")
