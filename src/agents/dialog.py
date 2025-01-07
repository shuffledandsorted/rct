"""Self-discovering command-line dialog system.

Commands marked with @command appear in the help menu.
All commands support argument parsing and state transitions.
"""

from typing import Dict, Any, Optional, Callable, TypeVar, cast
import inspect
import asyncio
import shlex
import sys
from functools import wraps

F = TypeVar("F", bound=Callable[..., Any])


def default(f: F) -> F:
    """Mark a method as the default command handler."""
    setattr(f, "_is_default", True)
    return f


def exit_command(f: F) -> F:
    """Mark a method to appear as the exit command."""
    setattr(f, "_is_exit", True)
    return f


def command(f: F) -> F:
    """Mark a method to appear in the help menu."""
    setattr(f, "_is_command", True)
    return f


def _parse_args(f: F, raw_args: list) -> dict:
    """Parse arguments according to function signature."""
    sig = inspect.signature(f)
    params = list(sig.parameters.values())[1:]  # Skip 'self'
    call_args = {}

    for i, param in enumerate(params):
        if i < len(raw_args):
            if param.kind == param.VAR_POSITIONAL:
                call_args[param.name] = raw_args[i:]
                break
            call_args[param.name] = raw_args[i]
        elif param.default != param.empty:
            continue
        elif param.kind == param.VAR_POSITIONAL:
            call_args[param.name] = []
        else:
            raise TypeError(f"Missing required argument: {param.name}")

    return call_args


class SelfDiscoveringDialog:
    def __init__(self):
        self._commands = self._discover_commands()
        self._default = self._discover_default()
        self._help_text = self._make_help_text()

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

    def _discover_default(self) -> Callable[[str], "SelfDiscoveringDialog"]:
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

    @command
    def help(self) -> "SelfDiscoveringDialog":
        """Display available commands."""
        print(self._help_text)
        return self

    @command
    def exit(self) -> None:
        """Exit the dialog system."""
        return None

    def process_command(self, user_input: str) -> Optional["SelfDiscoveringDialog"]:
        try:
            parts = shlex.split(user_input)
        except ValueError:
            parts = user_input.split()

        if not parts:
            return self

        cmd = parts[0].lower()
        args = parts[1:]

        # Try explicit command first
        if cmd in self._commands:
            method = getattr(self, self._commands[cmd]["method"])
            try:
                return method(**_parse_args(method, args))
            except Exception as e:
                print(f"Error: {e}")
                return self

        # Then try any public method
        if hasattr(self, cmd) and not cmd.startswith("_"):
            method = getattr(self, cmd)
            if callable(method):
                try:
                    return method(**_parse_args(method, args))
                except Exception as e:
                    print(f"Error: {e}")
                    return self

        # Fall back to default handler
        return self._default(user_input)

    def _fallback_command(self, user_input: str) -> "SelfDiscoveringDialog":
        print(f"Unknown command. Type 'help' for available commands.")
        return self


async def async_dialog(
    dialog: SelfDiscoveringDialog, prompt: str = "You: "
) -> SelfDiscoveringDialog | None:
    print(prompt, end="", flush=True)
    line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
    return dialog.process_command(line.rstrip("\n"))


async def run_async_dialog(
    dialog: SelfDiscoveringDialog, prompt: str = "You: "
) -> None:
    """Run the dialog system with the given prompt."""
    current = dialog
    try:
        while current is not None:
            try:
                await async_dialog(current, prompt)
            except Exception as e:
                print(f"Error: {e}")
    finally:
        print("\nDialog ended")
